# Real-world test images: sources

- **Dataset:** PlantDoc (Cropped-PlantDoc classification dataset), Singh et al., 2020, "PlantDoc: A Dataset for Visual Plant Disease Detection" (CODS-COMAD 2020). Photos are real in-field/web images, not lab-controlled PlantVillage captures.
- **Link:** https://github.com/pratikkayal/PlantDoc-Dataset
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0), per `LICENSE.txt` in that repository. Attribution: Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, Nipun Batra.
- **Split used:** the dataset's `test/` folder only. Up to 20 images per class were taken (first files in alphabetical order); most classes have fewer than 20 in the test split, so counts are 4-12 per class. Files are renamed `plantdoc_NN.<ext>`.

## Folder mapping (PlantDoc test folder -> class name used by the model)

| PlantDoc folder | real_images folder |
|---|---|
| Apple Scab Leaf | Apple___Apple_scab |
| Apple leaf | Apple___healthy |
| Apple rust leaf | Apple___Cedar_apple_rust |
| Bell_pepper leaf | Pepper,_bell___healthy |
| Blueberry leaf | Blueberry___healthy |
| Cherry leaf | Cherry_(including_sour)___healthy |
| Corn Gray leaf spot | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot |
| Corn rust leaf | Corn_(maize)___Common_rust_ |
| Peach leaf | Peach___healthy |
| Potato leaf early blight | Potato___Early_blight |
| Potato leaf late blight | Potato___Late_blight |
| Raspberry leaf | Raspberry___healthy |
| Soyabean leaf | Soybean___healthy |
| Squash Powdery mildew leaf | Squash___Powdery_mildew |
| Strawberry leaf | Strawberry___healthy |
| Tomato Early blight leaf | Tomato___Early_blight |
| Tomato Septoria leaf spot | Tomato___Septoria_leaf_spot |
| Tomato leaf bacterial spot | Tomato___Bacterial_spot |
| Tomato leaf late blight | Tomato___Late_blight |
| Tomato leaf mosaic virus | Tomato___Tomato_mosaic_virus |
| Tomato leaf yellow virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| Tomato leaf | Tomato___healthy |
| Tomato mold leaf | Tomato___Leaf_Mold |
| grape leaf black rot | Grape___Black_rot |
| grape leaf | Grape___healthy |

## Skipped (no clear match to a model class)

- `Bell_pepper leaf spot`: generic "leaf spot", not specifically bacterial spot.
- `Corn leaf blight`: not specified as Northern Leaf Blight.
- PlantDoc has no test folders for several model classes (e.g. Strawberry leaf scorch, Cherry powdery mildew, Grape leaf blight), so those are not covered.

"Healthy" folders are only as reliable as PlantDoc's own labels; the dataset was labelled by web-scraping plus human annotation and has some label noise.
