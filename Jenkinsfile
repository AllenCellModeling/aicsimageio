pipeline {
    parameters { booleanParam(name: 'create_release', defaultValue: false, 
                              description: 'If true, create a release artifact and publish to ' +
                                           'the artifactory release PyPi or public PyPi.') }
    options {
        timeout(time: 1, unit: 'HOURS')
    }
    agent {
        node {
            label "python-gradle"
        }
    }
    environment {
        PATH = "/home/jenkins/.local/bin:$PATH"
        REQUESTS_CA_BUNDLE = "/etc/ssl/certs"
    }
    stages {
        stage ("create virtualenv") {
            steps {
                this.notifyBB("INPROGRESS")
                sh "./gradlew -i cleanAll installCIDependencies"
            }
        }

        stage ("bump version pre-build") {
            when {
                expression { return params.create_release }
            }
            steps {
                // This will drop the dev suffix if we are releasing
                // X.Y.Z.devN -> X.Y.Z
                sh "./gradlew -i bumpVersionRelease"
            }
        }

        stage ("test/build distribution") {
            steps {
                sh "./gradlew -i build"
            }
        }

        stage ("report on tests") {
            steps {
                junit "build/test_report.xml"
                
                cobertura autoUpdateHealth: false,
                    autoUpdateStability: false,
                    coberturaReportFile: 'build/coverage.xml', 
                    failUnhealthy: false,
                    failUnstable: false,
                    maxNumberOfBuilds: 0,
                    onlyStable: false,
                    sourceEncoding: 'ASCII',
                    zoomCoverageChart: false
                

            }
        } 

        stage ("publish release") {
            when {
                branch 'master'
                expression { return params.create_release }
            }
            steps {
                sh "./gradlew -i publishRelease"
                sh "./gradlew -i gitTagCommitPush"
                sh "./gradlew -i bumpVersionPostRelease gitCommitPush"
             }
        }

        stage ("publish snapshot") {
            when {
                branch 'master'
                not { expression { return params.create_release } }
            }
            steps {
                sh "./gradlew -i publishSnapshot"
                script {
                    def ignoreAuthors = ["jenkins", "Jenkins User", "Jenkins Builder"]
                    if (!ignoreAuthors.contains(gitAuthor())) {
                        sh "./gradlew -i bumpVersionDev gitCommitPush"
                    }
                }
            }
        }

    }
    post {
        always {
            notifyBuildOnSlack(currentBuild.result, currentBuild.previousBuild?.result)
            this.notifyBB(currentBuild.result)
        }
        cleanup {
            deleteDir()
        }
    }
}

def notifyBB(String state) {
    // on success, result is null
    state = state ?: "SUCCESS"
    
    if (state == "SUCCESS" || state == "FAILURE") {
        currentBuild.result = state
    }

    notifyBitbucket commitSha1: "${GIT_COMMIT}", 
                credentialsId: 'aea50792-dda8-40e4-a683-79e8c83e72a6', 
                disableInprogressNotification: false, 
                considerUnstableAsSuccess: true, 
                ignoreUnverifiedSSLPeer: false,
                includeBuildNumberInKey: false, 
                prependParentProjectKey: false, 
                projectKey: 'SW', 
                stashServerBaseUrl: 'https://aicsbitbucket.corp.alleninstitute.org'
}

def notifyBuildOnSlack(String buildStatus = 'STARTED', String priorStatus) {
    // build status of null means successful
    buildStatus =  buildStatus ?: 'SUCCESS'

    // Override default values based on build status
    if (buildStatus != 'SUCCESS') {
        slackSend (
                color: '#FF0000',
                message: "${buildStatus}: '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})"
        )
    } else if (priorStatus != 'SUCCESS') {
        slackSend (
                color: '#00FF00',
                message: "BACK_TO_NORMAL: '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})"
        )
    }
}

def gitAuthor() {
    sh(returnStdout: true, script: 'git log -1 --format=%an').trim()
}
