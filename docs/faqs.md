# FAQs :question:

- Q1: System compatibility?
    - Ans: Reported successfull installation on different platforms include:
        - Windows 11: Python 3.8 with a virtual environment.
        - Ubuntu 22: Python 3.8, 3.9 and 3.10 with a virtual environment.

- Q2: Encountering issues with the JPype installation?
    - Ans: JPype seems to be not compatible with the most recent version of Python; check valid Python versions across platforms at Q1.
    
- Q3: Missing system-level dependencies on Linux?
    - Ans: Please ensure that the essential dev tools package has been deployed if you are using a Linux system. Also, according to [JPype's documentation](https://jpype.readthedocs.io/en/latest/install.html#debian-ubuntu), `g++` and `python-dev` need to be installed.

- Q4: `ModuleNotFoundError` error?
  - Ans: Please check there is no duplicated naming (e.g., `org`) in your paths because the Java dependencies would be overridden by that.