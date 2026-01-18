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

    # TODO: Answer the questions and fix math formating
    # TODO: Show every step of the process per rubric
    st.markdown(f"""
        a. Interpret dy/dx geometrically.\n
        b. How many differentiation formulas do we have and what are they?\n
        C. Diffrentiate the following:\n
            - y = 4 + 2x - 3x^2 - 5x^3 - 8x^4 + 9x^5\n
                Answer: dy/dx = 45x^4 - 32x^3 -15x^2 -6x +2\n
            - y = 1/x +3/x^2 +2/x^3\n
                Answer: dy/dx = -x^-2 - 6x^-3 - 6x^-4\n
            - y = ∛(3x^2)-1/√5x\n
                Answer: (2*3^(1/3) / 3*x(1/3)) + (5^(1/2) / 10*x^(3/2))\n
        d. Define partial derivative\n
        e. Given the following functions find dz/dx and dz/dy\n
            - z = 2x^2 -3xy + 4y^2\n
                Answer X: dz/dx = 4x - 3y\n
                Answer Y: dz/dy = 8y - 3x\n
            - z = x^2/y - y^2/x\n
                Answer X: dz/dx = (2x/y) + y^2*x^-2\n
                Answer Y: dz/dy = (-2*y/x) - (x^2/y^2)\n
            - z = e^(x^2+xy)\n
                Answer X: dz/dx = (2x+y) * e^(x^2 + x * y)\n
                Answer Y: dz/dy = x * e^(x^2 + x * y)\n
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