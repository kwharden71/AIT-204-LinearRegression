# ===========================
# IMPORTS
# ===========================
# Import streamlit for building interactive web interface
import streamlit as st
# Import pandas for data manipulation and analysis
import pandas as pd
# Import custom backend modules for data generation and linear regression
import backend 

# ===========================
# MAIN APPLICATION FUNCTION
# ===========================
def Main():
    # Configure the Streamlit page settings (title and layout)
    st.set_page_config(
        page_title="Linear Regression and Differential Equations",
        layout="wide"
    )

    # Display main title and description
    st.title("Linear Regression and Differential Equations")
    st.markdown("This application shows some basic Differential Equations as well discusses and show cases Linear Regression.")

    # ==============================
    # SECTION 1: DIFFERENTIAL EQUATIONS
    # ==============================
    # Display Part 1 - Differential Equations content and theory
    st.header("Part 1")

    st.markdown(f"""
        a. Interpret $\\frac{{dy}}{{dx}}$ geometrically.\n
        * $\\frac{{dy}}{{dx}}$ represents the first dervivative of an equation. Geometrically this is an equation that gives the slope of a tangent line of the observed equation.
                For example when we look at a linear line like y = 3x we know the tangent line to anypoint will always be y = 3. If we take the derviateve of y = 3x this gives us $\\frac{{dy}}{{dx}}$ = 3 which matches our expectation.
                This is very useful for a lot of reasons with one of the main ones being to find miniums or maximums as if we set the $\\frac{{dy}}{{dx}}$ equal to 0 and solve it gives us the values where the tangent line
                is flat which is only at minimum or maximums.

        b. How many differentiation formulas do we have and what are they?\n
        
        There are 28 formulas listed below:\n
        
        Basic Differentiations:\n
        |Function (y)|	Differentiation Formula (dy/dx)|
        |------------|---------------------------------|
        |c (constant)|	0|
        |$x^n$ (power)|	$nx^{{n-1}}$|
        |ln x (logarithmic)|	$\\frac{{1}}{{x}}$|
        |$e^x$(exponent)|	$e^x$|
        |$a^x$ (exponent)|	$a^x ln(a) $|
        
        Differentiation fo Trigonometric Functions:\n
        |Function (y)|	Derivative (dy/dx)|
        |------------|--------------------|
        |sin x|	cos x |
        |cos x|	-sin x |
        |tan x|	sec² x |
        |sec x|	sec x · tan x |
        |cosec x|	-cosec x · cot x |
        |cot x|	-csc² x |
                
        Differentation of Inverse Trigonometric Functions:\n
        |Function (y)|	Differentiation Formula (dy/dx)|
        | ------------|---------------------------------|
        |sin⁻¹ x|	1/√(1 - x²) |
        |cos⁻¹ x|	-1/√(1 - x²) |
        |tan⁻¹ x|	1/(1 + x²) |
        |sec⁻¹ x|	1/(\|x\|·√(x² - 1)) |
        |csc⁻¹ x|	-1/(\|x\|·√(x² - 1)) |
        |cot⁻¹ x|	-1/(1 + x²) |

        Differentation of Hyperbolic Functions:\n
        |Function (y)|	Differentiation Formula (dy/dx)|
        |--------|-------------|
        |sinh x |	cosh x|
        |cosh x |	sinh x|
        |tanh x |	sech² x|
        |sech x |	-sech x · tanh x|
        |cosech x |	-cosech x · coth x|
        |coth x |	-csch² x|
        
        Differentation Rules:\n
        |Rules|	Function Form (y) |	Differentiation Formula (dy/dx)|
        |-----|------------|-------------------------|
        |Sum Rule|	u(x) ± v(x) |	du/dx ± dv/dx |
        |Product Rule|	u(x) × v(x) |	u dv/dx + v du/dx |
        |Quotient Rule|	u(x) ÷ v(x) |	(v du/dx - u dv/dx) / v² |
        |Chain Rule|	f(g(x)) |	f'[g(x)] g'(x) |
        |Constant Rule|	k f(x), k ≠ 0 |	k d/dx f(x) |
        
        Reference: GeeksForGeeks (2025). Differentiation Formulas. geeksforgeeks.org. Retrived from https://www.geeksforgeeks.org/maths/differentiation-formulas/

        c. Differentiate the following:\n
                
        * $y = 4 + 2x - 3x^2 - 5x^3 - 8x^4 + 9x^5$\n
            
            $\\frac{{dy}}{{dx}} = 0 + 2 - 6x - 15x^2 - 32x^3 + 45x^4$\n
                
            * Answer: $\\frac{{dy}}{{dx}} = 45x^4 - 32x^3 - 15x^2 - 6x + 2$\n

        * $y = \\frac{{1}}{{x}} + \\frac{{3}}{{x^2}} + \\frac{{2}}{{x^3}}$\n
                
            $y = x^{{-1}} + 3x^{{-2}} + 2x^{{-3}}$\n
            
            $\\frac{{dy}}{{dx}} = -x^{{-2}} -6x^{{-3}} -6x^4$\n
                
            * Answer: $\\frac{{dy}}{{dx}} = -x^{{-2}} - 6x^{{-3}} - 6x^{{-4}}$\n

        * $y = \\sqrt[3]{{3x^2}} - \\frac{{1}}{{\\sqrt{{5x}}}}$\n
            
            $y = 3^{{1/3}}x^{{2/3}} - 5^{{-1/2}}x^{{-1/2}}$
            
            $\\frac{{dy}}{{dx}} = \\frac{{2 \\cdot 3^{{1/3}}}}{{3x^{{1/3}}}} + \\frac{{\\sqrt{{5}}}}{{10x^{{3/2}}}}$
            
            * Answer: $\\frac{{dy}}{{dx}} = \\frac{{2 \\cdot 3^{{1/3}}}}{{3x^{{1/3}}}} + \\frac{{\\sqrt{{5}}}}{{10x^{{3/2}}}}$\n

        d. Define partial derivative:\n
        * Derivatives work great for getting us the slope of the tangent line however we have a problem once we move beyond 2 Dimensions as they only work for 2D. This is where we introduce
        partial dervivatives as they allow us to break up the equation and take a derivative in respects to one of the dimensions while leaving the rest constant. For example
        lets say you have the equation y = 3x + 5z and we wanted to know the tangent slope. We can use partial deriviates to achieve this by first taking the derivative in respects to x while leaving z constant
        giving us dy/dx = 3 and then we can repreat for z and this time leaving x constant giving us dy/dz = 5.

        e. Given the following functions find $\\frac{{\\partial z}}{{\\partial x}}$ and $\\frac{{\\partial z}}{{\\partial y}}$\n

        * $z = 2x^2 - 3xy + 4y^2$
            
            $\\frac{{\\partial z}}{{\\partial x}} = 4x - 3y + 0$
                
            * Answer X: $\\frac{{\\partial z}}{{\\partial x}} = 4x - 3y$
            
            $\\frac{{\\partial z}}{{\\partial y}} = 0 - 3x + 8y$
            
            * Answer Y: $\\frac{{\\partial z}}{{\\partial y}} = 8y - 3x$

        * $z = \\frac{{x^2}}{{y}} - \\frac{{y^2}}{{x}}$
                
            $ z = x^2y^{{-1}}-y^2x^{{-1}}$
            
            $\\frac{{\\partial z}}{{\\partial x}} = 2xy^{{-1}} + y^{{2}}x^{{-2}}$
                
            * Answer X: $\\frac{{\\partial z}}{{\\partial x}} = \\frac{{2x}}{{y}} + \\frac{{y^2}}{{x^{{-2}}}}$
            
            $\\frac{{\\partial z}}{{\\partial y}} = -2yx^{{-1}} - x^2y^{{-2}}$
                
            * Answer Y: $\\frac{{\\partial z}}{{\\partial y}} = -\\frac{{2y}}{{x}} - \\frac{{x^2}}{{y^2}}$

        * $z = e^{{x^2 + xy}}$
                
            * Answer X: $\\frac{{\\partial z}}{{\\partial x}} = (2x + y)e^{{x^2 + xy}}$
                
            * Answer Y: $\\frac{{\\partial z}}{{\\partial y}} = xe^{{x^2 + xy}}$
    """)

    # ==============================
    # SECTION 2: LINEAR REGRESSION
    # ==============================
    # This section demonstrates building and training a linear regression model from scratch

    # ==========================================
    # SIDEBAR: DATA GENERATION PARAMETERS
    # ==========================================
    # Allow users to configure parameters for synthetic data generation
    st.sidebar.header("🎲 Data Generation")

    # Slider to control the number of synthetic data points to generate
    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=50,
        max_value=5000,
        value=1000,
        step=50,
        help="Number of data points to generate"
    )

    # Slider to control the amount of noise added to the synthetic data
    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Standard deviation of noise (higher = more noisy)"
    )

    # Input field for setting random seed to ensure reproducible results
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="For reproducible results"
    )

    st.sidebar.divider()

    # ==========================================
    # SIDEBAR: MODEL TRAINING PARAMETERS
    # ==========================================
    # Allow users to configure hyperparameters for the linear regression model
    st.sidebar.header("🤖 Model Parameters")

    # Slider to control learning rate (step size) for gradient descent optimization
    learning_rate = st.sidebar.slider(
        "Learning Rate (α)",
        min_value=0.0001,
        max_value=0.1,
        value=0.01,
        step=0.0001,
        format="%.4f",
        help="Step size for gradient descent"
    )

    # Slider to control the number of training iterations (epochs) for gradient descent
    n_iterations = st.sidebar.slider(
        "Number of Iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of training iterations"
    )

    # ===========================
    # DATA GENERATION
    # ===========================
    # Generate data button - triggers synthetic data generation when clicked
    if st.sidebar.button("🎲 Generate Data", type="primary"):
        # Create a data generator with the specified random seed for reproducibility
        generator = backend.data_generators.SyntheticDataGenerator(random_seed=random_seed)

        # Generate synthetic linear data with the specified number of samples and noise level
        # Store the generated data in Streamlit session state for persistence across reruns
        st.session_state.data = generator.generate_simple_linear(n_samples, noise_std= noise_level)
        st.session_state.data_generated = True

        # Display success message to the user
        st.sidebar.success("✅ Data generated!")

    # Check if data has been generated; if not, prompt user to generate data and stop execution
    if "data_generated" not in st.session_state:
        st.session_state.data_generated = False
        st.info("👆 Click 'Generate Data' to generate synthetic data")
        st.stop()

    # ===========================
    # MODEL TRAINING
    # ===========================
    # Retrieve the generated data from session state
    data = st.session_state.data
    
    # Split the data into training (80%) and testing (20%) sets
    train_df, test_df = backend.helpers.train_test_split(data, percentage=80)

    # Initialize the linear regression model with specified hyperparameters
    linear_model = backend.linear_regression.LinearRegression(
        learning_rate=learning_rate,
        n_iterations=n_iterations
    )

    # Train the model on the training set
    # X contains features (exclude 'y' and 'y_true' columns), y contains target values
    linear_model.fit(
        X=train_df.drop(columns=["y", "y_true"]),
        y=train_df["y"]
    )

    # Make predictions on the test set
    y_test_pred = linear_model.predict(test_df.drop(columns=["y", "y_true"]))

    # ===========================
    # RESULTS VISUALIZATION AND DISPLAY
    # ===========================
    # Display Part 2 header for Linear Regression section
    st.header("Part 2: Linear Regression from Scratch")

    # ===========================
    # Display Generated Dataset
    # ===========================
    st.subheader("Generated Data")
    st.write("First 10 rows of the generated dataset:")
    st.dataframe(data.head(10))
    st.write(f"Total samples: {len(data)}")
    st.write(f"Training samples: {len(train_df)} | Testing samples: {len(test_df)}")
    
    # Explain the data split strategy
    st.write("We opted to use an 80-20 train-test split for model evaluation.")

    # ===========================
    # Plot Regression Line
    # ===========================
    st.subheader("Scatter Plot of Data Points, Test Points, and Regression Line")
    linear_model.plot_regression_line(train_df, test_df)
    st.write("This is a scatter plot showcasing the generated data points (blue), test data points (orange), and the regression line (red) fitted by our linear regression model.")

    # ===========================
    # Training Set Performance Metrics
    # ===========================
    st.subheader("Model Performance on Training Set")
    # Get predictions on training data
    y_train_pred = linear_model.predict(train_df.drop(columns=["y", "y_true"]))
    # Calculate performance metrics on training set
    train_metrics = linear_model.compute_metrics(
        y_true=train_df["y"],
        y_pred=y_train_pred
    )
    # Display training metrics
    st.write("Training MSE:", f"{train_metrics['MSE']:.4f}")
    st.write("Training RMSE:", f"{train_metrics['RMSE']:.4f}")
    st.write("Training MAE:", f"{train_metrics['MAE']:.4f}")
    st.write("Training R²:", f"{train_metrics['R2']:.4f}")

    # ===========================
    # Test Set Performance Metrics
    # ===========================
    st.subheader("Model Performance on Test Set")
    # Calculate performance metrics on test set
    test_metrics = linear_model.compute_metrics(
        y_true=test_df["y"],
        y_pred=y_test_pred
    )
    # Display test metrics
    st.write("Mean Squared Error (MSE):", f"{test_metrics['MSE']:.4f}")
    st.write("Root Mean Squared Error (RMSE):", f"{test_metrics['RMSE']:.4f}")
    st.write("Mean Absolute Error (MAE):", f"{test_metrics['MAE']:.4f}")
    st.write("R-squared (R²):", f"{test_metrics['R2']:.4f}")
    
    # Explain the significance of the metrics
    st.write("These metrics indicate how well our linear regression model performed on unseen test data. A high R² value (close to 1) suggests a good fit, meaning the model explains a large portion of the variance in the data. Lower values of MSE, RMSE, and MAE indicate better predictive accuracy." \
    " Overall, these results demonstrate the effectiveness of our linear regression implementation.")

    # ===========================
    # Display Learned Model Parameters
    # ===========================
    st.subheader("Model Parameters")
    st.write("Weights:", linear_model.weights[0])
    st.write("Bias:", linear_model.bias)
    st.write("""These are the parameters obtained from training the linear regression model using gradient descent. Our base equation is given by y = Xw + b where w represents the weights and b represents the bias.
             The weights indicate the influence of each feature on the target variable, while the bias allows the model to fit the data better by providing an offset.
             Our synthetic equation was generated using y = 2x + 1 + ε and we can compare the obtained weights and bias to see how close they are to the original parameters used in data generation.""")

    # ===========================
    # Display Training Progress
    # ===========================
    st.subheader("Loss Curve")
    # Plot the MSE loss over training iterations
    st.plotly_chart(linear_model.plot_training_progress())
    st.write("The loss curve illustrates how the Mean Squared Error (MSE) decreased over the gradient descent iterations. A declining loss indicates that the model is learning effectively and converging towards an optimal solution.")

    # ===========================
    # Display Residuals Analysis
    # ===========================
    st.subheader("Residuals Plot of Test Set")
    # Plot the residuals (differences between actual and predicted values)
    st.plotly_chart(linear_model.plot_residuals(test_df["y"], y_test_pred))
    st.write("The residuals plot displays the differences between the actual and predicted values on the test set. Ideally, the residuals should be randomly distributed around zero, indicating that the model has captured the underlying patterns in the data without systematic bias.")
    
    # ===========================
    # Ethical Considerations
    # ===========================
    st.subheader("Ethical Considerations")
    st.write("When deploying machine learning models in real-world applications, it is important to consider ethical implications such as fairness, transparency, and accountability. Our linear regression model is a simple example and does not account for potential biases in the data or model predictions.")
    st.write("In practice, one should ensure that the training data is representative of the population, avoid using sensitive attributes that could lead to discrimination, and provide explanations for model predictions to stakeholders. Regular audits and updates to the model may also be necessary to maintain ethical standards over time.")


# ===========================
# APPLICATION ENTRY POINT
# ===========================
# Run the main application when the script is executed directly
if __name__ == "__main__":
    Main()