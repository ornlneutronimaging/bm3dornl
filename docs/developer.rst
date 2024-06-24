===============
Developer Guide
===============

.. contents::
    :local:


This guide is intended for developers who want to contribute to the development of the project.

-----------------
Developer Account
-----------------

All development work is done on Github and requires a Github account.
For developers who are part of SCSE@ORNL, you need to make sure you have the 2-step verification enabled on your account to comply with the organization's security policy.
For developers who are not part of SCSE@ORNL, you need to fork the repository to your own account and submit pull requests to the main repository.

--------
Branches
--------

The default branch for all development work is the `next` branch, and all pull requests should be submitted to this branch.
Review is required for all pull requests, and the pull request will be merged into the `next` branch after the review is complete by the project maintainers.
In addition to the `next` branch, we also have `qa` and `main` branches.
The `qa` branch is used for testing the release candidate, and the `main` branch is used for the stable release.
All three branches are protected branches, and only the project maintainers can merge into the `main` branch.

-----------------
Development Cycle
-----------------

For developers who are part of SCSE@ORNL, the development cycle is as follows:

* Create a new branch from the `next` branch for the new feature or bug fix.
* Make changes to the code, adding and updating tests as needed.
* Submit a pull request to the `next` branch, and assign the pull request to the project maintainers for review.

   * The pull request should have a short and meaningful title to describe the changes.
   * A link to the internal issue tracker should be included in the pull request description, and the pull request link should be included in the issue tracker. (This step will become automatic once we have the issue tracker integrated with Github.)
   * Provide test instructions if needed.
   * Provide example output if needed.
   * Keep in mind that we might need to come back to the pull request later, so make sure to provide enough information for the reviewers and future developers to understand the changes.

* The project maintainers will review the pull request and provide feedback.

For developers who are not part of SCSE@ORNL, the development cycle is as follows:

* Open an issue to describe the new feature or bug fix.
* Fork the repository to your own account.
* Create a new branch from the `next` branch for the new feature or bug fix.
* Make changes to the code, adding and updating tests as needed.
* Submit a pull request to the `next` branch, and assign the pull request to the project maintainers for review.

   * The pull request should have a short and meaningful title to describe the changes.
   * Mention the issue number in the pull request description, such as "this pull request resolves issue #123."
   * Provide test instructions if needed.
   * Provide example output if needed.
   * Keep in mind that we might need to come back to the pull request later, so make sure to provide enough information for the reviewers and future developers to understand the changes.

* The project maintainers will review the pull request and provide feedback.

-------------
Release Cycle
-------------

The release cycle is as follows:

* The project maintainers will create a release candidate from the `next` branch.
* The release candidate will be tested on the `qa` branch.
* If the release candidate passes all tests, it will be merged into the `main` branch.
* The `main` branch will be tagged with the release version number.
* The release version will be published on Github, Anaconda and PyPI.

-------------------------
Developing with Test Data
-------------------------

To develop with the provided integration test data, please follow the instructions below:

  * Clone the repository to your local machine.
  * Make sure `git lfs` is installed on your machine.
  * Run `git lfs install` to enable `git lfs` for the repository.
  * Run `git submodule init` to initialize the submodule.
  * Run `git submodule update` to fetch the submodule data.

Now you should have the test data available in the `tests/bm3dornl-data` directory.

If you would like to add small testing data, you can use the `tests/data` directory.
Please make sure to keep the data small and do not commit large data files to the repository.
