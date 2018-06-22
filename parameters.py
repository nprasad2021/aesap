class Experiment(object):

    def __init__(self, identity=0, name='base', precursor='/Users/i869533/Documents/aesap/', datatype = 'open_data'):
        
        # Indexing
        self.ID = identity
        self.overname = name
        self.name = 'ID' + str(self.ID) + "_" + name
        self.category = 'initial/'

        self.reuse_TFRecords = True
        self.tdata = datatype

        ### Directory Details
        self.log_dir_base = "log/"
        self.fig_dir_base = 'fig/'
        self.results = 'evaluation.txt'
        self.accuracy = 'accuracy.txt'
        self.precursor = precursor # Path to root


        self.keep = 1 # Models to keep
        self.K = 10 # Similarity Score, show _ queries
        self.similarity_distance = 'cosine'
        self.classification = False
        self.vae = True

        self.save_every = 1000
        self.decay_every = 1000
        self.learning_rate = .0005
        self.decaying_rate = 1.0
        self.loss = 'l2'
        self.build = 1
        self.num_units = 150

        self.mode = 'both'
        self.train_dir = self.precursor + 'data/' + self.tdata +  '/train/' # Training Directory
        self.val_dir = self.precursor + 'data/' + self.tdata + '/val/' # Validation Directory
        self.outfile = 'finalresults.txt'

        #### Hyperparameters
        self.batch_size = 128
        self.image_size = 224
        self.num_epochs = 100
        self.scale = 128
        self.slide = 0

        self.extense_summary = True
        self.restart = True

        self.pipeline = self.precursor + self.log_dir_base + self.category + self.name
        self.figline = self.precursor + self.overname + '_' + self.fig_dir_base + self.category + self.name + '/'
        self.resultline = self.precursor + self.overname + '_' + self.results
        self.accline = self.precursor + self.overname + '_' + self.accuracy
        self.tfr_out = self.precursor + 'data/' + self.tdata + '/records/' # TFRecords Output Path

    def __str__(self):
        a = '--------Model Parameters----- \n'
        a += 'Name: ' + str(self.name) + '\n'
        a += 'Image Size: ' + str(self.image_size) + '\n'
        a += 'Batch Size: ' + str(self.batch_size) + '\n' 
        a += 'Dataset: ' + self.tdata + '\n'
        a += 'LR: ' + str(self.learning_rate) + '\n'
        a += 'Epochs: ' + str(self.num_epochs) + '\n'
        a += 'Scale: ' + str(self.scale) + '\n' 
        a += 'Slide: ' + str(self.slide) + '\n' 
        a += 'Build: ' + str(self.build) + '\n'
        a += 'Hidden Units: ' + str(self.num_units)
        a += 'Pipeline: ' + self.pipeline + '\n' 
        a += '-----------------------------------'
        return a



