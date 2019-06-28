from tensorflow import keras


class DataGenerator():
    def __init__(self, ptrain, ptest=None, augmentation=True, validation_split=0.2):
        """Data generation and augmentation
        # Arguments
            ptrain: string, training data folder .
        """
        self.ptrain = ptrain
        if not ptest:
            self.ptest = ptrain
            self.train_subset = "training"
            self.valid_subset = "validation"
            self.validation_split = validation_split
        else:
            self.ptest = ptest
            self.train_subset = ''
            self.valid_subset = ''
            self.validation_split = 0.0

        if augmentation:
            self.data_gen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1. / 255,
                rotation_range = 30,
                shear_range=0.5,
                horizontal_flip=True,
                zoom_range=0.4,
                validation_split=self.validation_split)
        else:
            self.data_gen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1. / 255,
                validation_split=self.validation_split)


    def generate(self, batch, size):
        train_generator = self.data_gen.flow_from_directory(
            self.ptrain,
            target_size=(size, size),
            batch_size=batch,
            class_mode='categorical',
            shuffle = True,
            subset = self.train_subset)

        validation_generator = self.data_gen.flow_from_directory(
            self.ptest,
            target_size=(size, size),
            batch_size=batch,
            class_mode='categorical',
            shuffle = True,
            subset = self.valid_subset)


        return train_generator, validation_generator
