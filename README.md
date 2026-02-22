# ai-dataset-creator
creates headshots from videos

roject Overview

This is a Streamlit-based web application that:

• Uploads a video file
• Reduces redundant frames based on similarity threshold
• Detects faces using two methods:

Haar Cascade (Fast)

DNN SSD (More Accurate)
• Displays detected faces from:

Original video

Processed video
• Allows downloading processed video parts
• Automatically splits large videos (>200MB)
