import h5py

# Open the HDF5 file in read-only mode
with h5py.File('test_data.h5', 'r') as file:
    # Loop through each sentence group
    for sentence_name in file.keys():
        sentence_group = file[sentence_name]
        print(f"Sentence: {sentence_name}")

        # Access the 'words' dataset in the sentence group
        words_dataset = sentence_group.keys()

        # Loop through each word in the dataset
        for word_name in words_dataset:
            
            print(sentence_name, "*****", word_name)
            print(file[sentence_name][word_name].attrs['vector'])