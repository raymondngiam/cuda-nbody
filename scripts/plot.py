from matplotlib import pyplot as plt
import numpy as np
names = ['V0-cpu_baseline', 'V1-RefactorBodyForce', 'V2-KernelIntegration', 'V3-Prefetch', 'V4-SharedMemory', 'V5-SharedMemoryMultiThread']
values = [0.038, 19.006683, 35.673443, 39.940559, 63.897747, 75.182198]

f,a=plt.subplots(figsize=(10, 10))
a.set_xlabel('Configurations')
a.set_xticklabels(names,rotation=90)
a.set_xlabel('Configurations')
a.set_ylabel('Billon Interactions / Sec')
a.set_title('N-Body Simulation Perfomance (N=4096)')
a.grid(True)
a.plot(names,values,'b*-')

plt.subplots_adjust(top=0.88,
bottom=0.34,
left=0.11,
right=0.9,
hspace=0.2,
wspace=0.2)
plt.savefig('../images/performance.png')
plt.show()
