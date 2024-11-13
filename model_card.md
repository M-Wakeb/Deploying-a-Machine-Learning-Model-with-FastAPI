# Model Card

For additional information see the Model Card paper: [Paper](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
This model created by M-
Model uses `RandomForest` from `sklearn` for classification.
no HyperParameters

## Intended Use
Model predicts if a person earnings are over 50K or not. 
## Training Data
Features:
- age: age in years
- workclass: work category (e.g., private, government...etc)
- fnlwgt: 
- education: highest level of education completed 
- education-num: education level as numeric 
- marital-status: marital status
- occupation: type of job
- relationship: family role of the person 
- race: racial group (e.g., White, Asian ...etc)
- sex: gender
- capital-gain: investment income 
- capital-loss: investment loss
- hours-per-week: avg
- native-country: country of birth
Gategorical features are encodded using `OneHotEncoder`. 
## Evaluation Data
The dataset is first preprocessed then split into training and evaluation into 70/30%.
## Metrics
Metrics used and model's performance.
- `Precision: 0.71`
- `Recall: 0.61 `
- `F-beta: 0.66`
## Ethical Considerations
This model is trained on census data which can be found here:
[Data](https://archive.ics.uci.edu/dataset/20/census+income)
## Caveats and Recommendations
I recommend to continuously monitor and validate the model's predictions in real-world applications. 