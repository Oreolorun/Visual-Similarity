from helper import *

st.title('Visual Similarity Recommendation')

st.image('images/header_image.jpg', use_column_width=True)

st.success(
    """
    This app takes in an image of a car and recommends 4 similar images from a bunch of car images which I have randomly 
    curated. Typically the model will try to find similar images by chassis, if there are none then it tries to find 
    images with similar backgrounds and object orientation.

    ###### Note:
    Although this app has some logic to handle adversarial attacks, it is still advised to only upload images of cars as
    the underlying model is only intended for that use case and adversarial images (images not of cars) will only result 
    in wrong recommendations as expected. 
    """
)

st.write('#### Upload Section')

uploaded = st.file_uploader('Upload image here', type=['jpg', 'jpeg', 'png'])

output(uploaded=uploaded)
