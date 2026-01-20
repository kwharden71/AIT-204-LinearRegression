import streamlit as st
import pandas as pd
import backend 

def Main():
    st.set_page_config(
        page_title="Linear Regression and Differential Equations",
        layout="wide"
    )

    st.title("Linear Regression and Differential Equations")
    st.markdown("This application shows some basic Differential Equations as well discusses and show cases Linear Regression.")

    # =====================================
    # SECTION 1: Part 1
    # =====================================
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
    # SECTION 2: Part 2
    # ==============================

    st.sidebar.header("🎲 Data Generation")

    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Number of data points to generate"
    )

    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Standard deviation of noise (higher = more noisy)"
    )

    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="For reproducible results"
    )

    st.sidebar.divider()

    # ==========================================
    # SIDEBAR - Model Parameters
    # ==========================================

    st.sidebar.header("🤖 Model Parameters")

    learning_rate = st.sidebar.slider(
        "Learning Rate (α)",
        min_value=0.0001,
        max_value=0.1,
        value=0.01,
        step=0.0001,
        format="%.4f",
        help="Step size for gradient descent"
    )

    n_iterations = st.sidebar.slider(
        "Number of Iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of training iterations"
    )

    # Generate data button
    if st.sidebar.button("🎲 Generate Data", type="primary"):
        generator = backend.data_generators.SyntheticDataGenerator(random_seed=random_seed)

        # Store in session state
        st.session_state.data = generator.generate_simple_linear(n_samples, noise_std= noise_level)
        st.session_state.data_generated = True

        st.sidebar.success("✅ Data generated!")

    st.markdown(f"{st.session_state.data.head()}")

if __name__ == "__main__":
    Main()