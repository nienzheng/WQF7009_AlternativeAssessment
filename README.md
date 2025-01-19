1. XAI_DASHBOARD.py is dashboard built for alternative assignment part 2 designed using Questions driven XAI design. It consists of the following sections:
   - Sidebar: upload chest x-ray image to get pneumonia prediction
   - Model prediction: prediction and 2 XAI explanations: Grad-CAM and LIME
   - Model performance: performance details of model
   - Data: view the first 30 images of Test/Validation/Train datasets
   - About Dataset: dataset description
   - About Model: model description
2. WQF7009 Alternative Assessment Part 1 - Code.py is the code built for alternative assignment part 2 to train 3 models: Logistic regression, ResNet-18, VGG-16
   - Executing the code will train the 3 models and the trained model weights of VGG-16 (the best performer) will be stored locally as "vgg16_trained_model_2.pth" which can be used for XAI_DASHBOARD.py

To run XAI_DASHBOARD.py which built using streamlit, start Anaconda Terminal, and enter commands below:
  cd <filepath of XAI_DASHBOARD.py>
  streamlit run XAI_DASHBOARD.py

NOTE 1: Further information for execution is available in: https://docs.streamlit.io/develop/concepts/architecture/run-your-app
NOTE 2: The streamlit app will try to load the weights from "vgg16_trained_model_2.pth" which located in the same folder, if the file is unavailable, the pretrained weight of VGG16 model will be used. "vgg16_trained_model_2.pth" can be obtained by running the code "WQF7009 Alternative Assessment Part 1 - Code.py"

