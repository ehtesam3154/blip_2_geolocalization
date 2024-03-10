Unfortunately the dataset that we used here is not public (at least we do not have the permission to share it publicly, but the dataset had almost 13.5k images and each images were labelled city_name_some_extension and there was a csv file that napped each city to ots state, country, continent)

Here I used BLIP'2 original framework from https://github.com/salesforce/LAVIS to test it for zero-shot geo-localization tasks with Image-to-Text retreival and visual question answering (open and close-ended vqa)

Also tested with hybrid vqa approach (first open-ended vqa then close-ended) and retrieval-first-then-vqa approach (:

Just paste all the code in the LAVIS folder after you clone LAVIS in your local machine or server
