import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered",
)


st.title("🌸 Iris Flower Classification")
st.write(
    "Enter the flower measurements below, then click **Predict Flower**."
)


sepal_length = st.number_input(
    "Sepal length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1,
)

sepal_width = st.number_input(
    "Sepal width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1,
)

petal_length = st.number_input(
    "Petal length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1,
)

petal_width = st.number_input(
    "Petal width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1,
)


if st.button("Predict Flower"):
    input_data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        response = requests.post(
            API_URL,
            json=input_data,
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]
            probabilities = result["probabilities"]

            st.success(
                f"Predicted flower: {prediction.capitalize()}"
            )

            st.subheader("Prediction probabilities")

            for flower_name, probability in probabilities.items():
                st.write(
                    f"{flower_name.capitalize()}: "
                    f"{probability * 100:.2f}%"
                )
                st.progress(float(probability))

        else:
            st.error(
                f"API error: {response.status_code}"
            )
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend. "
            "Make sure the API is running on port 8000."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The API request took too long."
        )

    except requests.exceptions.RequestException as error:
        st.error(
            f"Request failed: {error}"
        )