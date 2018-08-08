#!groovy

node ("python-gradle")
{
    parameters { booleanParam(name: 'promote_artifact', defaultValue: false, description: '') }
    def is_promote=(params.promote_artifact)
    echo "BUILDTYPE: " + (is_promote ? "Promote Image" : "Build, Publish and Tag")

    try {
        def VENV_BIN = "/local1/virtualenvs/jenkinstools/bin"
        def PYTHON = "${VENV_BIN}/python3"

        stage ("git") {
            def git_url=gitUrl()
            if (env.BRANCH_NAME == null) {
                git url: "${git_url}"
            }
            else {
                println "*** BRANCH ${env.BRANCH_NAME}"
                git url: "${git_url}", branch: "${env.BRANCH_NAME}"
            }
        }

        if (!is_promote) {
            stage ("prepare version") {
                sh "${PYTHON} ${VENV_BIN}/manage_version -t python -s prepare"
            }

            stage("build and publish") {
                sh './gradlew -i cleanAll publish'
            }

            stage ("tag and commit") {
                sh "${PYTHON} ${VENV_BIN}/manage_version -t python -s tag"
            }

            junit "build/test_report.xml"
        }
        else {
            stage("promote") {
                sh "${PYTHON} ${VENV_BIN}/promote_artifact -t python -g ${params.git_tag}"
            }
        }

        currentBuild.result = "SUCCESS"
    }
    catch(e) {
        // If there was an exception thrown, the build failed
        currentBuild.result = "FAILURE"
        throw e
    }
    finally {

        if (currentBuild?.result) {
            println "BUILD: ${currentBuild.result}"
        }
        // Slack
        notifyBuildOnSlack(currentBuild.result, currentBuild.previousBuild?.result)

        // Email
        step([$class: 'Mailer',
              notifyEveryUnstableBuild: true,
              recipients: '!AICS_DevOps@alleninstitute.org',
              sendToIndividuals: true])
    }
}

def gitUrl() {
    checkout scm
    sh(returnStdout: true, script: 'git config remote.origin.url').trim()
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
