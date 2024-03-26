import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import csv
import pyttsx3
import re
from sklearn.model_selection import train_test_split

# Load data
training = pd.read_csv('Data/Training.csv')
cols = training.columns[:-1]

# Load dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}

with open('MasterData/symptom_severity.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if len(row) >= 2:
            severityDictionary[row[0]] = int(row[1])

with open('MasterData/symptom_Description.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if len(row) >= 2:
            description_list[row[0]] = row[1]

with open('MasterData/symptom_precaution.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if len(row) >= 5:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# Train decision tree classifier
x = training[cols]
y = training['prognosis']
clf = DecisionTreeClassifier()
clf.fit(x, y)


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names):


# Implementation of tree_to_code function


# Streamlit app
def main():
    st.title("HealthCare ChatBot")

    st.markdown("---")

    st.write("Please enter your name:")
    user_name = st.text_input("You:", "")

    if user_name:
        st.write(f"Hello, {user_name}!")

        st.write("Now, please enter the symptom you are experiencing:")

        user_symptom = st.text_input("You:", "")

        if user_symptom:
            st.write("Are you experiencing any other symptoms related to this? (yes/no)")

            additional_symptoms = st.radio("You:", ["yes", "no"])

            if additional_symptoms == "yes":
                # Ask for additional symptoms
                pass
            else:
                # Make prediction based on provided symptom
                conf, cnf_dis = check_pattern(cols, user_symptom)
                if conf == 1:
                    st.write("Related symptoms found:")
                    for num, it in enumerate(cnf_dis):
                        st.write(f"{num}) {it}")
                    if num != 0:
                        conf_inp = st.number_input(f"Select the one you meant (0 - {num}):", min_value=0, max_value=num)
                        disease_input = cnf_dis[conf_inp]
                    else:
                        disease_input = cnf_dis[0]
                else:
                    st.write("Enter valid symptom.")

                while True:
                    try:
                        num_days = int(st.text_input("Okay. From how many days?"))
                        break
                    except:
                        st.write("Enter valid input.")

                present_disease = sec_predict([disease_input])
                st.write("Are you experiencing any ")
                symptoms_exp = []
                for syms in list(reduced_data.columns[reduced_data.loc[present_disease].values[0].nonzero()]):
                    inp = st.radio(syms + "?", ["yes", "no"])
                    if inp == "yes":
                        symptoms_exp.append(syms)

                calc_condition(symptoms_exp, num_days)
                if present_disease[0] in description_list:
                    st.write(f"You may have {present_disease[0]}")
                    st.write(description_list[present_disease[0]])
                else:
                    st.write(f"Unknown disease: {present_disease[0]}")

                if present_disease[0] in precautionDictionary:
                    st.write("Take following measures:")
                    for i, j in enumerate(precautionDictionary[present_disease[0]]):
                        st.write(f"{i + 1}) {j}")

    st.markdown("---")


if __name__ == "__main__":
    main()
