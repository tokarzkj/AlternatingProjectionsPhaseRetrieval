# AlternatingProjectionsPhaseRetrieval

This repos was created for the purpose of determining if we can blindly recover a signal from an image.
The codebase relies on PyQT to provide a user interface to help people run the experiments that have been
used to solve the above problem. This library makes use of numpy, scipy, and other libraries to help perform
calculations as fast as possible.

### Benchmarks
This project uses richbench to execute a variety of tests to confirm the best way to handle our most rigorous
calculations. The benchmarks can be found in the benchmark folder, and you can run them by executing 
```richbench benchmarks/```, where benchmarks/ is the target folder.

### Tests
This project uses pytest to make sure changes in our algorithms result in equivalent changes. As a result,
not all tests are meant to detect regressions at all times.


### Contributors
The code is primarily written by Kris Tokarz with assistance and supervision from Aditya Viswanathan for
an Independent Study in Winter 2023 at the University of Michigan - Dearborn.
