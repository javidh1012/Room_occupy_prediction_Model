import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
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
INTEGER_STEP_FEATURES = {
	"S1_Light",
	"S2_Light",
	"S3_Light",
	"S4_Light",
	"S5_CO2",
	"S6_PIR",
	"S7_PIR",
}
DEFAULT_SENSOR_READINGS = {
	"S1_Temp": 25.81,
	"S2_Temp": 26.88,
	"S3_Temp": 25.50,
	"S4_Temp": 26.25,
	"S1_Light": 12,
	"S2_Light": 13,
	"S3_Light": 62,
	"S4_Light": 45,
	"S1_Sound": 0.09,
	"S2_Sound": 0.08,
	"S3_Sound": 0.09,
	"S4_Sound": 0.08,
	"S5_CO2": 535,
	"S5_CO2_Slope": 0.438462,
	"S6_PIR": 0,
	"S7_PIR": 0,
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


def build_feature_frame(user_inputs, model_features):
	"""Build a single-row DataFrame matching the trained PyCaret pipeline schema."""
	feature_row = {feature: np.nan for feature in model_features}
	for name in USER_INPUT_FEATURES:
		if name in user_inputs:
			feature_row[name] = float(user_inputs[name])
	return pd.DataFrame([feature_row], columns=model_features)


def get_neighbor_diagnostics(trained_model, feature_frame):
	"""Return transformed features and nearest-neighbor details for KNN pipelines."""
	steps = getattr(trained_model, "steps", None)
	if not steps:
		return None, None
	estimator = steps[-1][-1]
	if not hasattr(estimator, "kneighbors"):
		return None, None
	transformed = trained_model[:-1].transform(feature_frame) if len(steps) > 1 else feature_frame
	distances, indices = estimator.kneighbors(transformed)
	classes = getattr(estimator, "classes_", None)
	raw_labels = getattr(estimator, "_y", None)
	neighbors = []
	for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
		label = None
		if raw_labels is not None:
			raw_val = raw_labels[idx]
			label = int(classes[raw_val]) if classes is not None and raw_val < len(classes) else int(raw_val)
		neighbors.append({"Neighbor": rank, "Train Row Index": int(idx), "Euclidean Distance": round(float(dist), 4), "Neighbor Class": label})
	return transformed, pd.DataFrame(neighbors)


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
		default_value = DEFAULT_SENSOR_READINGS.get(feature_name, 0.0)
		min_value = -100.0 if feature_name == "S5_CO2_Slope" else 0.0
		step = 1.0 if feature_name in INTEGER_STEP_FEATURES else 0.01
		number_format = "%.0f" if feature_name in INTEGER_STEP_FEATURES else "%.4f"
		label = feature_name.replace("_", " ")
		inputs[feature_name] = column.number_input(
			label,
			min_value=min_value,
			value=float(default_value),
			step=step,
			format=number_format,
			key=f"input_{feature_name}",
		)
	submitted = st.form_submit_button("Predict occupancy", type="primary")

if submitted:
	feature_frame = build_feature_frame(inputs, MODEL_FEATURES)
	try:
		prediction = trained_model.predict(feature_frame)
		predicted_value = prediction[0]
		formatted_prediction = (
			f"{float(predicted_value):g}"
			if isinstance(predicted_value, (int, float, np.number))
			else str(predicted_value)
		)
		st.metric("Predicted room occupancy", formatted_prediction)

		if hasattr(trained_model, "predict_proba"):
			probas = trained_model.predict_proba(feature_frame)[0]
			classes = getattr(trained_model, "classes_", list(range(len(probas))))
			proba_text = " | ".join(f"Class {cls}: {prob:.0%}" for cls, prob in zip(classes, probas))
			st.caption(f"Class probabilities — {proba_text}")

		with st.expander("Prediction debugging info"):
			st.write("Input DataFrame passed to `trained_model.predict(...)` (column order matches `MODEL_FEATURES`):")
			st.dataframe(feature_frame, hide_index=True)
			transformed, neighbors_df = get_neighbor_diagnostics(trained_model, feature_frame)
			if neighbors_df is not None:
				st.write("Nearest training neighbors (unscaled Euclidean distance):")
				st.dataframe(neighbors_df, hide_index=True)
				st.caption(
					"Note: This PyCaret pipeline uses KNeighborsClassifier (k=5) without feature scaling. "
					"High-range features (S5_CO2, S1_Light–S4_Light) dominate Euclidean distance, so changing "
					"only small-scale features (Temp, Sound, PIR, CO2 Slope) while leaving Light/CO2 near "
					"empty-room levels can keep the nearest neighbors in Class 0."
				)
			if transformed is not None:
				st.write("Transformed pipeline features:")
				st.dataframe(transformed, hide_index=True)
	except Exception as error:
		st.error("Prediction failed. Confirm the input features and model version match the trained model.")
		with st.expander("Technical details"):
			st.code(str(error))
