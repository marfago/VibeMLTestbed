# Vibe ML Testbed Developer Guide

## Way of Working

The following guidelines should be followed when developing this project:

*   Each project operation must be managed/run by poetry. 
*   Code files must not be longer than 1000 lines.
*   Code must be modularized and well architected.
*   Code must be adhere to the separation of concerns principle.
*   Tests must all pass and run quicly. 
*   90% test coverege must be lawes reached.
*   User stories must be fully tested with unit tests and 95% coverage.
*   After a new unit test is added, all the tests must run successfully.
*   The user stories status must be tracked in a specific file.
*   A user guide with instruction for a user to set up/use the code must be written.
*   After a user story is implemented, it must be committed.
*   Commit messages should be of the form "Implement user story #[number]: [user story description]".
*   After a user story is completed, the user story status must be updated following the pattern
        US_Id. US_Description: US_Status