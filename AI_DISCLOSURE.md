# AI Usage Disclosure

## AI Tool Used

- ChatGPT (OpenAI)

## How It Was Used and Impact

1. **Conceptual Clarification**
   - Used AI alongside the assignment and course materials to understand FastAPI validation and metrics, load testing with Little's Law, MLflow models and aliases, promotion gates, shadow testing, Ray Serve, stable hashing and release smoke tests.
   - **Impact:** Helped me understand how the four parts connect and interpret my results.

2. Code Generation
- Used GPT to help with some parts of the implementation and tests in partA/ through partD/  have cited the excat references inside the files it self with doc strings at the start of each file. ,Ex: using the course’s Lecture 4a–4c serving exercises as references but the codes from it were not directly used , I modified them with GPT and made teh currect versions. The supplied eta_data.py and load_test.py were unchanged.
- With AI assistance, I changed the integer-field validation after a valid CSV-derived request was rejected: pandas had represented integer-valued inputs as floats. I removed strict type enforcement while retaining the integer fields, range limits and infeasible-delivery rule. The valid request then returned HTTP 200, while invalid requests still returned HTTP 422.
- Impact: Helped adapt the implementation to the supplied data while preserving the required validation behavior.

3. Debugging and Troubleshooting
- Used AI to investigate problems encountered during execution. For example, Ray Serve initially failed to serialize the FastAPI application with a cannot pickle '_thread.lock' object error. Installing older FastAPI and Starlette versions in my main environment then caused a Streamlit dependency conflict. I restored the main environment and installed the compatible serving packages separately under .local/partC/serve-deps, selecting them through PYTHONPATH for Part C. I reran the smoke check and verified that normal requests reached v1 and the X-Beta: 1 override reached v2.
- Impact: Helped resolve the compatibility issue while preserving the dependencies needed by my other work, and enabled the Ray Serve verification to proceed.

4. **Report, Command and Documentation Assistance**
   - Used AI to explain terminal commands and help draft and revise `README.md`, report explanations and results tables using my saved outputs and the rubric's requested topics.
   - **Impact:** Helped organize the reproduction instructions and present the results and limitations within the four-page limit.

## Responsibility 

- I have reviewed and the  ran the services, tests and experiments on my laptop MAC in side the VritualBox(Linux aarch64 VM) with 4 logical CPUs, 7.6 GiB RAM and Python 3.10.13 for the original measurements.
- The screenshots, logs and measured results all comes from actual executions that i have run and the vedio record can act as a proof for that and The report identifies untested behavior and proposed release criteria.
- I acknowledge that I have used  AI assistance to help with the code and debugging , tests and documentation reformating and remain responsible for the submission's correctness and for explaining, for modifying the codes when there were errors and troubleshooting the code.
-  I have checked that every number in `REPORT.pdf` matches the output files in `partA/` to `partD/

## References

- [DA3408 course repository](https://github.com/sudarsun/DA3408): assignment, supplied starter files and Lecture 4a–4c serving exercises (notebook filenames begin with `Lecture 10`).
- [OpenAI ChatGPT](https://chatgpt.com).
