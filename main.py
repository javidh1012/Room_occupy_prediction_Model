from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().with_name("Room_Occupancy_prediction.joblib")
USER_INPUT_FEATURES = [
	"S1_Temp",
	"S2_Temp",
	"S3_Temp",
	"S4_Temp",
	"S1_Light",
	"S2_Light",
	"S3_Light",
	"S4_Light",
	"S1_Sound",
	"S2_Sound",
	"S3_Sound",
	"S4_Sound",
	"S5_CO2",
	"S5_CO2_Slope",
	"S6_PIR",
	"S7_PIR",
]
DEFAULT_MEANS = {
	"S1_Temp": 25.4558026,
	"S2_Temp": 25.5489866,
	"S3_Temp": 25.0588027,
	"S4_Temp": 25.7564505,
	"S1_Light": 25.5531285,
	"S2_Light": 26.1282241,
	"S3_Light": 34.5755893,
	"S4_Light": 13.1883253,
	"S1_Sound": 0.167779834,
	"S2_Sound": 0.119434777,
	"S3_Sound": 0.16038134,
	"S4_Sound": 0.104535358,
	"S5_CO2": 462.203505,
	"S5_CO2_Slope": -0.0141353158,
	"S6_PIR": 0.0887325682,
	"S7_PIR": 0.0792299148,
}


def get_model_object(model_obj):
	if isinstance(model_obj, dict):
		return model_obj.get("model", model_obj)
	return model_obj


def get_model_features(model_obj):
	trained_model = get_model_object(model_obj)
	feature_names = getattr(trained_model, "feature_names_in_", None)
	if feature_names is not None:
		return list(feature_names)
	if isinstance(model_obj, dict):
		return [feature for feature in model_obj.get("features", []) if feature != "Room_Occupancy_Count"]
	return USER_INPUT_FEATURES


st.set_page_config(page_title="Room Occupancy Prediction", page_icon=":material/meeting_room:")


@st.cache_resource
def load_model():
	return joblib.load(MODEL_PATH)


st.title("Room Occupancy Prediction")
st.write("Enter the sensor readings to estimate the room occupancy count.")

if not MODEL_PATH.is_file():
	st.error(f"Model file not found: {MODEL_PATH.name}. Place it beside main.py.")
	st.stop()

try:
	model = load_model()
except ModuleNotFoundError as error:
	missing_module = error.name or "a required module"
	st.error(
		f"Could not load the model because {missing_module} is not installed. "
		"Install the packages in requirements.txt. PyCaret models may also require "
		"the same PyCaret version used when the model was saved."
	)
	st.stop()
except Exception as error:
	st.error("The model could not be loaded. Check that it is a valid, compatible joblib file.")
	with st.expander("Technical details"):
		st.code(str(error))
	st.stop()

trained_model = get_model_object(model)
MODEL_FEATURES = [feature for feature in get_model_features(model) if feature != "Room_Occupancy_Count"]

with st.form("occupancy_inputs"):
	st.subheader("Sensor readings")
	cols = st.columns(4)
	inputs = {}
	for index, feature_name in enumerate(USER_INPUT_FEATURES):
		column = cols[index % 4]
		default_value = DEFAULT_MEANS.get(feature_name, 0.0)
		min_value = -100.0 if feature_name == "S5_CO2_Slope" else 0.0
		label = feature_name.replace("_", " ")
		inputs[feature_name] = column.number_input(
			label,
			min_value=min_value,
			value=float(default_value),
			step=0.01,
			format="%.4f",
		)
	submitted = st.form_submit_button("Predict occupancy", type="primary")

if submitted:
	feature_row = {"Date": "", "Time": ""}
	for name in USER_INPUT_FEATURES:
		feature_row[name] = inputs[name]
	feature_frame = pd.DataFrame([feature_row], columns=MODEL_FEATURES)
	try:
		prediction = trained_model.predict(feature_frame)
		predicted_value = prediction[0]
		st.metric("Predicted room occupancy", f"{predicted_value:g}" if isinstance(predicted_value, (int, float)) else str(predicted_value))
	except Exception as error:
		st.error("Prediction failed. Confirm the input features and model version match the trained model.")
		with st.expander("Technical details"):
			st.code(str(error))
