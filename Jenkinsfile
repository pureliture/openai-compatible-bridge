pipeline {
    agent {
        kubernetes {
            label 'openai-compatible-bridge-builder'
            yaml """
spec:
  nodeSelector:
    neurons.openclaw.io/local-registry: "true"
  containers:
  - name: python
    image: python:3.12-slim
    command: ['sleep']
    args: ['infinity']
  - name: docker
    image: docker:27-cli
    command: ['sleep']
    args: ['infinity']
    securityContext:
      runAsGroup: 973
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  - name: git-tools
    image: alpine/git:latest
    command: ['sleep']
    args: ['infinity']
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
"""
        }
    }

    environment {
        REGISTRY = 'localhost:5000'
        IMAGE_NAME = 'neurons/openai-compatible-bridge'
        GITOPS_REPO = 'https://github.com/pureliture/neurons-ops.git'
        GITOPS_BRANCH = 'main'
        GITOPS_MANIFEST = 'k3s/neurons/overlays/production/model-gateway-bridge.yaml'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git config --global --add safe.directory "$WORKSPACE"'
                script {
                    env.GIT_SHORT = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.IMAGE_FULL = "${env.REGISTRY}/${env.IMAGE_NAME}:sha-${env.GIT_SHORT}"
                    env.IMAGE_LATEST = "${env.REGISTRY}/${env.IMAGE_NAME}:latest"
                }
            }
        }

        stage('Test') {
            steps {
                container('python') {
                    sh '''
                        set -eu
                        python -m pip install --no-cache-dir uv
                        uv sync
                        uv run pytest -q
                        python -m compileall -q openai_compatible_bridge
                    '''
                }
            }
        }

        stage('Docker Build & Push') {
            steps {
                container('docker') {
                    sh '''
                        set -eu
                        docker build -t "$IMAGE_FULL" -t "$IMAGE_LATEST" .
                        docker push "$IMAGE_FULL"
                        docker push "$IMAGE_LATEST"
                    '''
                }
            }
        }

        stage('GitOps Update') {
            steps {
                container('git-tools') {
                    withCredentials([usernamePassword(credentialsId: 'github-pat', usernameVariable: 'GIT_USER', passwordVariable: 'GIT_TOKEN')]) {
                        sh '''
                            set -eu
                            rm -rf /tmp/neurons-ops
                            git clone --branch "$GITOPS_BRANCH" --single-branch "https://${GIT_USER}:${GIT_TOKEN}@github.com/pureliture/neurons-ops.git" /tmp/neurons-ops
                            cd /tmp/neurons-ops
                            git config --global --add safe.directory /tmp/neurons-ops
                            git config user.email "jenkins@k3s-master-01"
                            git config user.name "Jenkins CI"
                            if ! grep -Eq 'localhost:5000/neurons/openai-compatible-bridge:sha-[A-Za-z0-9._-]+' "$GITOPS_MANIFEST"; then
                                echo "openai-compatible-bridge image tag pattern not found"
                                exit 1
                            fi
                            current_port=$(grep -E 'containerPort: 1893[12]' "$GITOPS_MANIFEST" | head -1 | grep -Eo '1893[12]')
                            if [ "$current_port" = "18931" ]; then
                                next_port="18932"
                            else
                                next_port="18931"
                            fi
                            sed -i -E "s|localhost:5000/neurons/openai-compatible-bridge:sha-[A-Za-z0-9._-]+|${IMAGE_FULL}|g" "$GITOPS_MANIFEST"
                            sed -i -E "s|\"1893[12]\"|\"${next_port}\"|g; s|containerPort: 1893[12]|containerPort: ${next_port}|g" "$GITOPS_MANIFEST"
                            git add "$GITOPS_MANIFEST"
                            if git diff --cached --quiet; then
                                echo "GitOps manifest already up to date: $IMAGE_FULL"
                            else
                                git commit -m "ci: update openai-compatible-bridge image ${IMAGE_FULL}"
                                git push origin "HEAD:${GITOPS_BRANCH}"
                            fi
                        '''
                    }
                }
            }
        }
    }
}
