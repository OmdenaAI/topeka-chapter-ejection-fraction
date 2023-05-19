def generate_data(filelist, img_path, gt_df):
    while True:
        for i in filelist:
            if i.endswith(".avi"):
                cap = cv2.VideoCapture(img_path + '/' + i)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                img = np.array(frames)[:, :, :, 0]
                img = img[:28]
                resized_img = np.zeros((28, 112, 112))
                for j, k in enumerate(img):
                    resized_img[j, :, :] = cv2.resize(k, (112, 112), interpolation=cv2.INTER_LINEAR) / 255
                y = round(gt_df.EF[np.where(gt_df.FileName == i[:-4])[0][0]])

                spatial_data = resized_img[:, :, :56]
                temporal_data = np.zeros((27, 112, 56))
                for j in range(1, 28):
                 
                    subtracted_frame = resized_img[j, :, 56:] - resized_img[j - 1, :, 56:]
                    std = np.std(subtracted_frame)
                    if std == 0:
                        std = 1e-6 # set a small epsilon value if standard deviation is zero
                    normalized_frame = (subtracted_frame - np.mean(subtracted_frame)) / std
                    temporal_data[j - 1, :, :] = normalized_frame
                   

                yield spatial_data[:, :, :, np.newaxis], temporal_data[:, :, :, np.newaxis], y

                
def two_stream_batch_generator(batch_size, gen_x):
    while True:
        spatial_batch_data = np.empty((batch_size, 28, 112, 56, 1))
        temporal_batch_data = np.empty((batch_size, 27, 112, 56, 1))
        batch_labels = np.empty((batch_size, 1))

        for i in range(batch_size):
            spatial_data, temporal_data, label = next(gen_x)
            spatial_batch_data[i] = spatial_data
            temporal_batch_data[i] = temporal_data
            batch_labels[i] = label  # Ensure the labels have the shape (batch_size, 1)

        yield [spatial_batch_data, temporal_batch_data], batch_labels
