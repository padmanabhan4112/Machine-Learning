# Machine Learning
ML Theory and Python/PyTorch codes

# 1. Linear Regression

<img width="725" height="525" alt="image" src="https://github.com/user-attachments/assets/6738319c-a6a5-42a2-b0ed-c5943e530da5" />

Best Fit line to the data using -> [theta1 (weight1) = The slope] && [theta0 (bias) = The y interscept / staring value of slope].

Sum square error:
<img width="725" height="556" alt="image" src="https://github.com/user-attachments/assets/0b1c741a-e644-4726-8f4f-91c6bd995225" />

Why Sum square error?
1. To perform differentiation easily in optimization step.
2. To eliminate the negative values when distance is calculated.
3. Becomes sensitive to outliers - The squared error makes the model terrified of being far away from any single point. It forces the line to "listen" more to the points that are furthest away. This is a disadvantage with sum sq error.

<img width="725" height="516" alt="image" src="https://github.com/user-attachments/assets/a9672f08-d29a-4d60-919d-3983fb5ced28" />

Curve Fitting: (Nonlinear in the data and linear in the parameters)
<img width="725" height="555" alt="image" src="https://github.com/user-attachments/assets/d9dd02a8-2b72-42aa-a87d-ef314170924b" />

The Optimization part of Linear Regression:
<img width="725" height="501" alt="image" src="https://github.com/user-attachments/assets/dea2bd13-8800-4d06-861a-cd1c5d7f4b4c" />

Our objective would be to minimize the loss function and find optimal weights -> theta0 and theta1 for the minimized loss function.

Pytorch Implementation  - Forward pass for loss computing & Backward pass for gradeint calculation:
<img width="725" height="741" alt="image" src="https://github.com/user-attachments/assets/e9d54c2f-aef7-46c3-9d3e-9b647393f27a" />

Solution part after applying derivatives to the loss function:
<img width="725" height="541" alt="image" src="https://github.com/user-attachments/assets/d90a02fd-e842-451d-be6f-da5edbf15344" />

Solving Linear Regression using Matrix-Vector form (Another way of solving Linear Regression problem):
<img width="725" height="535" alt="image" src="https://github.com/user-attachments/assets/c88588bd-f278-4cf8-8969-fb7e6e7eeb61" />
<img width="725" height="499" alt="image" src="https://github.com/user-attachments/assets/394d7d06-ce98-466f-ba1a-7891724a1f74" />

Quardratic Fitting (Even tough the curve is not straight we can apply weights which act as simple multipliers rather than doing exponential or sine functions):

<img width="725" height="500" alt="image" src="https://github.com/user-attachments/assets/527bfb4a-f463-434a-b418-47e50d7ce337" />
<img width="725" height="525" alt="image" src="https://github.com/user-attachments/assets/8e6a10e1-3f06-46b8-a82c-b2247ecf5640" />

Legendre polynomials (Orthogonal polynomials):
<img width="725" height="545" alt="image" src="https://github.com/user-attachments/assets/8fc09aa2-adf1-4741-bcad-64dbabcfc79f" />
<img width="725" height="546" alt="image" src="https://github.com/user-attachments/assets/61fd873a-6aad-4ddf-abcf-2620dd611455" />

While using higher order terms -> x = 0.99, then x^{10} ~0.904 and x^{11} ~0.895 -> To a computer trying to invert a matrix, those two columns look so similar that it starts to think they are the same column.
Because Legendre polynomials are Orthogonal, the matrix X^T*X becomes a Diagonal Matrix (or very close to it).

# 2. Under determined & Over determined System

<img width="725" height="526" alt="image" src="https://github.com/user-attachments/assets/5c757707-b1ae-421b-bb67-f7b73a4464b7" />

Our Loss will be minimum only if y =/~ x*theta. Therefore the solution to the system theta can have unique, 
infinitely many non-unique solutions, or no solution.

Some useful math concept to visualize:

<img width="725" height="530" alt="image" src="https://github.com/user-attachments/assets/445c982c-308f-4a60-b576-20063ecc4fe3" />
<img width="725" height="518" alt="image" src="https://github.com/user-attachments/assets/959af297-e780-4bd5-ba84-7c35cb015de6" />
<img width="725" height="531" alt="image" src="https://github.com/user-attachments/assets/be83a51e-01bc-497d-a1a3-db50bbd0a91b" />

Over determined System & Under determined System:
(Note - Visualize Over determined System & Under determined System's loss function has a bowl shaped curve either with trough or unique global minima).

Over determined System:

<img width="725" height="405" alt="image" src="https://github.com/user-attachments/assets/ca2ab425-1450-4465-9615-ff3d987d51df" />

Under determined System:

<img width="725" height="533" alt="image" src="https://github.com/user-attachments/assets/b21a5097-c026-4d2d-9fd4-939a44d164ad" />

P3: The "Minimum Norm" Solution: This specific formula represents the minimum norm solution. 
Among all the infinite possible solutions that exist in an under-determined system, 
this specific $\hat{\theta}$ is the unique one that is closest to the origin (the shortest vector).



