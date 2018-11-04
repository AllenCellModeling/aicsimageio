#!groovy

node ("python-gradle")
{
    parameters { booleanParam(name: 'create_release', defaultValue: false, 
                              description: 'If true, create a release artifact and publish to ' +
                                           'the artifactory release PyPi or public PyPi.') }
    def create_release=(params.create_release)
    echo "BUILDTYPE: " + (create_release ? "Creating a Release" : "Building a Snapshot")

    try {
        stage ("git pull") {
            def git_url=gitUrl()
            if (env.BRANCH_NAME == null) {
                git url: "${git_url}", branch: "master"
            }
            else {
                println "*** BRANCH ${env.BRANCH_NAME}"
                git url: "${git_url}", branch: "${env.BRANCH_NAME}"
            }
        }

        stage ("initialize virtualenv") {
            sh "./gradlew -i cleanAll installCIDependencies"
        }

        stage ("bump version pre-build") {
            // This will drop the dev suffix if we are releasing
            if (create_release) {
                // X.Y.Z.devN -> X.Y.Z
                sh "./gradlew -i bumpVersionRelease"
            }
        }

        stage ("test/build distribution") {
            sh './gradlew -i build'
        }

        junit "build/test_report.xml"
        step([$class: 'CoberturaPublisher', autoUpdateHealth: false, autoUpdateStability: false,
              coberturaReportFile: 'build/coverage.xml', failUnhealthy: false, failUnstable: false,
              maxNumberOfBuilds: 0, onlyStable: false, sourceEncoding: 'ASCII', zoomCoverageChart: false])

        stage ("publish") {
            def publish_task = create_release ? "publishRelease" : "publishSnapshot"
            sh "./gradlew -i ${publish_task}"
        }

        stage ("tag release") {
            if (create_release) {
                sh "./gradlew -i gitTagCommitPush"
            }
            else {
                println "This is a snapshot build - it will not be tagged."
            }
        }

        stage ("prep for dev") {
            if (create_release) {
                // X.Y.Z -> X.Y.Z+1.dev0  (default - increment patch)
                sh "./gradlew -i bumpVersionPostRelease gitCommitPush"
            }
            else {  // This is a snapshot build
                // X.Y.Z.devN -> X.Y.Z.devN+1  (devbuild)
                def ignoreAuthors = ["jenkins", "Jenkins User", "Jenkins Builder"]
                if (!ignoreAuthors.contains(gitAuthor())) {
                    sh "./gradlew -i bumpVersionDev gitCommitPush"
                }
                else {
                    println "This is a snapshot build from a jenkins commit. The version will not be bumped."
                }
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
    //checkout scm
    sh(returnStdout: true, script: 'git config remote.origin.url').trim()
}

def gitAuthor() {
    //checkout scm
    sh(returnStdout: true, script: 'git log -1 --format=%an').trim()
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
