
import parameters

#### PARAMETERS
# Final TUNING
dsts = ['all_data', 'open_data', 'sap_data']
lrs = [.0005, .0001]
img_sizes = [192, 144]
scale = [1]
slide = [0]
bs = [128, 64, 32]
build = [3,2,1]
num_units = [200, 150, 100, 50]
vae = [True, False]


def gen_tune_exp(precursor):

	opt = []
	idx = 0
	print(precursor)
	# PARAMETERIZE BASELINE
	opt += [parameters.Experiment(identity=idx, precursor=precursor)]

	idx += 1
	for dst in dsts:
		for sl in slide:
			for sc in scale:
				for b in bs:
					for lr in lrs:
						for imsize in img_sizes:
							for bill in build:
								for hidden in num_units:
									for v in vae:


										name = 'modelK'
										opt += [parameters.Experiment(identity=idx, name=name, precursor=precursor, datatype=dst)]
										opt[-1].image_size = imsize
										opt[-1].learning_rate = lr
										opt[-1].outfile = 'finalresults.txt'
										opt[-1].scale = sc
										opt[-1].slide = sl
										opt[-1].batch_size = b
										opt[-1].build = bill
										opt[-1].restart = True
										opt[-1].mode = 'both'
										opt[-1].num_units = hidden
										opt[-1].category = 'visual/'
										opt[-1].vae = v
										opt[-1].loss = 'l2'

										idx += 1

	print('tune experiments num:', len(opt))
	return opt

if __name__ == '__main__':
	gen_tune_exp('dfaasdfadsf')