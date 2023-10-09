import numpy as np

def rle(data, binary = False, val_bits = None, len_bits = None):

	# Flatten the data
	data = data.flatten()

	# Calculate val_bits if not given
	if binary:
		val_bits = 0
	elif val_bits is None:

		# Get lowerbound
		min_val = np.min(data)
		if min_val < 0:
			min_val = int(np.ceil(np.log2(-min_val)) + 1)
		elif min_val == 0:
			min_val = 1
		else:
			min_val = int(np.floor(np.log2(min_val)) + 1)

		# Get upperbound
		max_val = np.max(data)
		if max_val < 0:
			max_val = int(np.ceil(np.log2(-max_val)) + 1)
		elif max_val == 0:
			max_val = 1
		else:
			max_val = int(np.floor(np.log2(max_val)) + 1)

		val_bits = np.maximum(min_val, max_val)

	bitstream = [[data[0], 1]]
	max_seq = 1
	for i in range(1, len(data)):
		if data[i] == bitstream[-1][0]:
			bitstream[-1][1] += 1
			if bitstream[-1][1] > max_seq:
				max_seq = bitstream[-1][1]
		else:
			bitstream.append([data[i], 1])

	max_seq = np.floor(np.log2(max_seq)) + 1
	if len_bits is None or len_bits < max_seq:
		len_bits = int(max_seq)

        '''
	output = ''
	for i in range(len(bitstream)):
		val = ''
		if not binary:
			val = np.binary_repr(bitstream[i][0], width = val_bits)
		length = np.binary_repr(bitstream[i][1], width = len_bits)

		if i == 0 and bitstream[i][0] == 1:
			val = np.binary_repr(0, width = len_bits)

		output += val + length
        '''

	return np.array(bitstream), val_bits, len_bits

if __name__ == '__main__':
	data = (np.random.rand(10,10) < 0.1) * 1
	print(data)
	print(rle(data, binary = True))
