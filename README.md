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

Our Loss will be minimum only if y = ~(x*theta). Therefore the solution to the system theta can have unique, 
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

P3: The "Minimum Norm" Solution - This specific formula represents the minimum norm solution. 
Among all the infinite possible solutions that exist in an under-determined system, 
this specific $\hat{\theta}$ is the unique one that is closest to the origin (the shortest vector).

# 3. Regularization

Regularization is a technique used to prevent overfitting. 
Regularization is a technique used in machine learning to improve model stability and prevent overfitting by penalizing the magnitude of the model's coefficients.

<img width="725" height="542" alt="image" src="https://github.com/user-attachments/assets/8693f658-64bd-4b43-bbad-5035b802c7fc" />

L2 Regularization:
L2 regularization (also known as Ridge regression) is a method used to prevent model instability by penalizing the magnitude of the weights using a squared constraint.

<img width="725" height="530" alt="image" src="https://github.com/user-attachments/assets/d238f167-fa98-4566-b823-f0e877d97ed7" />

Why squared constraint?
1. Easy to differentiate.
2. The L2 Regularization term measures the "length" or magnitude of the weights.
3. By adding this to the loss function, the model is forced to keep weights small because large weights now "cost" more in the total error calculation.

<img width="725" height="534" alt="image" src="https://github.com/user-attachments/assets/e7892675-6d7a-4f1f-8ff4-97b6c912fa03" />

Note: The loss minimization is calcuated with lamda value using trial and error.

<img width="725" height="538" alt="image" src="https://github.com/user-attachments/assets/40597ca8-f498-4925-ba9f-c2d5862411a7" />

<img width="725" height="517" alt="image" src="https://github.com/user-attachments/assets/95467d52-a694-4e10-a53c-a6071ba1c2f0" />

Note: The constraint is represented by the circle and loss function by an ellipsoidal shape. 
The global minimum for the ellipsoidal is at the center represented by red cross but after regularization the green cross is final optimal loss solution.

L1 Regularization (Lasso Regression):
L1 regularization adds a penalty to the loss function based on the absolute values of the model's coefficients. It discourages the model from relying on too many features, often pushing the weights of unimportant variables exactly to zero.
First the loss function and required weights are computed and later the respective data values and traget values are compare with the penalty lambda, if the respective compared values are not in range of penalty defined then the weights are squased to zero. 

Intuition of L1 Regularization: 
The Squared Error part of the loss is calculated and to minimize this with respect to a single weight theta_j, we take the derivative wrt to theta_j.
Equating the derivative to zero, we get theta_j. theta_j is equated as y_j. Shrinkage operation is applied to y_j and the not potential/if in the penalty range y_j are squashed to zero.

<img width="725" height="651" alt="image" src="https://github.com/user-attachments/assets/bcfc8a57-a5fa-4c3c-ba79-1870a88c97c1" />
<img width="725" height="342" alt="image" src="https://github.com/user-attachments/assets/06351bdb-3dfa-4be9-a1ff-99665a56d6f7" />
<img width="725" height="643" alt="image" src="https://github.com/user-attachments/assets/4af7e60b-0072-4022-9ced-9cb53ea1e267" />

<img width="725" height="530" alt="image" src="https://github.com/user-attachments/assets/13a1d2d8-6da9-43ad-ad61-b739a427661a" />

Geometric Interpretation (L1 Regularization):
Mostly the constraint is optimized at the corners thus reducing the Weights/Features.
<img width="725" height="531" alt="image" src="https://github.com/user-attachments/assets/98a366bf-7ec8-4cc4-a915-1a763005de4c" />

<img width="725" height="521" alt="image" src="https://github.com/user-attachments/assets/4629b547-7d84-4aed-a5c9-836f6d738224" />

Note: Here A is considered as I (Identity matrix) for easy solving. Usually the loss is computed first and later the shrinkage operator is applied.

<img width="725" height="524" alt="image" src="https://github.com/user-attachments/assets/77a7a2e2-3058-4b3b-a96b-375e41158e9e" />

# 4. Kernal Method

In Kernel method, instead of picking a line / a quadratic equation, we pick a kernel.
A kernel is a measure of distance between training samples.
Kernel method buys us the ability to handle nonlinearity.
Ordinary regression is based on the columns (features) of A. Kernel method is based on the rows (samples) of A.

<img width="725" height="517" alt="image" src="https://github.com/user-attachments/assets/d56431cb-b4da-44ee-8273-e6fcf9826e24" />

In Kernel Method we introduce a gaussian curve to each data points. Using Kernel function we find the correlation between given test data point and existing trained data points (scalar value).
To achieve this, we require to build the K matrix using the training data. Each entry of the K matrix K_{ij} is the Gaussian correlation between training point i and training point j, 
so that we can compute the alpha coefficients to find the new expected output value. Finally for the test data, we multiply the alpha coefficients with Kernel function (scalar value) we get the expected output value.
Note: We can convert the Ordinary regression formula which is based on the columns (features) of A into the Kernel method is based on the rows (samples) of A.
<img width="725" height="347" alt="image" src="https://github.com/user-attachments/assets/cb59fa30-7e84-4e7d-89c4-145ba468f1a8" />

<img width="725" height="526" alt="image" src="https://github.com/user-attachments/assets/eae61486-6a21-43be-be17-34071c00eb7d" />

<img width="725" height="529" alt="image" src="https://github.com/user-attachments/assets/00c32438-d906-4bf5-a8d7-507016ac8451" />
Note: We can convert the Ordinary regression which is based on the columns (features) of A formulation into the Kernel method is based on the rows (samples) of A.

<img width="725" height="538" alt="image" src="https://github.com/user-attachments/assets/80c22292-04c2-413d-92c1-ddc2617be6ae" />

<img width="725" height="518" alt="image" src="https://github.com/user-attachments/assets/0d32e12a-481f-41bf-9d6b-2ff246706b1e" />
Note: This is how a Kernel fucntion is defined.

<img width="725" height="486" alt="image" src="https://github.com/user-attachments/assets/1f5357a7-64dd-4b54-b59a-9afdc5a9fe3a" />
Note : A*A(T) is replaced by Kernel matrix K.  

<img width="725" height="529" alt="image" src="https://github.com/user-attachments/assets/9761ada6-2ca0-4f2a-a3a9-a6fdcaadd805" />

<img width="725" height="527" alt="image" src="https://github.com/user-attachments/assets/ed87db7b-4aa5-4684-946e-9abd10ccb5a1" />



