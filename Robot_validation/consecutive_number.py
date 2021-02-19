
queue = [2,3, 6, 7, 10, 11, 12, 15, 16, 17, 21, 22, 30]

packet = []

tmp = []

v = queue.pop(0)

tmp.append(v)

print(v)

while(len(queue)>0):

	vv = queue.pop(0)

	print(vv)

	if v+1 == vv:

		tmp.append(vv)

		v = vv

	else:

		packet.append(tmp)

		tmp = []

		tmp.append(vv)

		v = vv



packet.append(tmp)



print(packet)
