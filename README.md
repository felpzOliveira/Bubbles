# Bubbles

![Alt text](/images/fluid.png)

This is my personal fluid solver, it is heavily based on the [fluid-engine-development](https://fluidenginedevelopment.org/) book by Doyub Kim, but with GPU support using CUDA - I don't however make any claims of it's performance and quality.

Currently it contain particle-based solvers for the following methods:
* SPH - Smoothed Particle Hydrodynamics
* PCI-SPH - Predict-Corrective SPH
* ESPIC - Eletrostatic PIC

This code was also used for my master's thesis and the code used for boundary routines is located under src/boundaries, and currently supports:
* Color Field [MCG03]
* Asymmetry [HLW+12]
* Sandim's Method [SCN+16, SPd20]
* Dilts [Dill00, HD07]
* Randles-Doring/Marrone - partially [MCLTG10]

Building this code is straightforward, simply use the given CMakeLists and make the project. You will need a CUDA capable device (and nvcc installed) and Qhull (for Samdim's Method). If you don't want to build with Qhull, you can set 'CONVEXHULL\_QHULL' to OFF when building. It will trigger an internal implementation of the Quick Hull algorithm, however the quality of Sandim's routine will suffer.
