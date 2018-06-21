
import parameters

#### PARAMETERS
# Final TUNING
dsts = ['sap_data', 'all_data', 'open_data']
lrs = [.005, .001, .0005, .0001]
img_sizes = [112, 144, 192, 224]
scale = [1,32,64,128]
slide = [0, 64, -64]
loss = ['l2', 'l1']
bs = [32, 64, 128]
build = [2,1]


def gen_tune_exp(precursor):

	opt = []
	idx = 0

	# PARAMETERIZE BASELINE
	opt += [parameters.Experiment(identity=idx, precursor=precursor)]

	idx += 1
	for dst in dsts:
		for sl in slide:
			for sc in scale:
				for b in bs:
					for lr in lrs:
						for imsize in img_sizes:
							for l in loss:
								for bill in build:

									name = 'tune_lr' #+ str(lr) + '_' + dst + '_' + str(imsize) + '_sc' + str(sc) + '_sl' + str(sl) + '_bui' + str(bill) + '_l' + str(l) + '_bs' + str(b)
									opt += [parameters.Experiment(identity=idx, name=name, precursor=precursor, datatype=dst)]
									opt[-1].image_size = imsize
									opt[-1].learning_rate = lr
									opt[-1].outfile = 'finalresults.txt'
									opt[-1].scale = sc
									opt[-1].slide = sl
									opt[-1].loss = l
									opt[-1].batch_size = b
									opt[-1].build = bill
									opt[-1].restart = True

									idx += 1

	print('tune experiments num:', len(opt))
	return opt

if __name__ == '__main__':
	gen_tune_exp('dfaasdfadsf')