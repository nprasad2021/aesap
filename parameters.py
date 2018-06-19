class Experiment(object):

    def __init__(self, identity=0, name='base', precursor='/Users/i869533/Documents/aesap/', datatype = 'sap_data'):
        
        # Indexing
        self.ID = identity
        self.name = 'ID' + str(self.ID) + "_" + name
        self.category = 'initial'
        self.gpu = 2

        self.reuse_TFRecords = False
        self.tdata = datatype

        ### Directory Details
        self.log_dir_base = "log/"
        self.precursor = precursor # Path to root
        self.tfr_out = self.precursor + 'data/' + self.tdata + '/records/' # TFRecords Output Path

        self.keep = 1 # Models to keep
        self.K = 10 # Similarity Score, show _ queries
        self.similarity_distance = 'cosine'
        self.classification = False

        self.save_every = 1000
        self.decay_every = 1000
        self.learning_rate = .0005
        self.decaying_rate = 1.0

        self.mode = 'train'
        self.train_dir = self.precursor + 'data/' + self.tdata +  '/train/' # Training Directory
        self.val_dir = self.precursor + 'data/' + self.tdata + '/val/' # Validation Directory

        #### Hyperparameters
        self.batch_size = 128
        self.image_size = 224
        self.num_epochs = 20

        self.extense_summary = True
        self.restart = True


