import numpy as np
import matplotlib.pyplot as plt
import dsp

# generating a signal
n = 500
a = np.empty([n])
f = np.empty([n])
for i in range(n):
    f[i] = (500 + i + 1) / n
    factor_a  = np.random.rand()
    a[i] = (n / (i + 1) * factor_a) ** (1/5) * 10
plt.figure('Test Signal Frequence')
plt.plot(f,a)
plt.savefig('Test_Signal_Frequence_Domain.png', dpi = 600)
plt.show()

t_signal = np.arange(0,20,0.0001)
y_signal = np.empty(len(t_signal))
for i in range(n):
    y_i = a[i] * np.sin(2*np.pi*f[i]*t_signal + np.random.rand()*2*np.pi)
    y_signal = y_signal + y_i
plt.figure('Test Signal Time')
plt.plot(t_signal, y_signal)
plt.savefig('Test_Signal_Time_Domain.png', dpi = 600)
plt.show()

Y_signal = np.array([[y_signal],[t_signal]])

fa = 10
Y_sampling = dsp.sampling(Y_signal, fa)
y_sampling = Y_sampling[0,:].flatten()
t_sampling = Y_sampling[1,:].flatten()
plt.figure('Sampled Signal')
plt.plot(t_signal, y_signal)
plt.plot(t_sampling, y_sampling, '.')
plt.savefig('Sampled_Signal.png', dpi = 600)
plt.show()

Y_dft = dsp.dft(Y_sampling)
f_dft = Y_dft[1,:].flatten()
a_dft = Y_dft[0,:].flatten()
plt.figure('DFT')
plt.plot(f_dft, abs(a_dft),'.')
plt.savefig('DFT', dpi = 600)
plt.show()

Y_idft = dsp.idft(Y_dft)
t_idft = Y_idft[1,:].flatten()
y_idft = Y_idft[0,:].flatten()
plt.figure('IDFT')
plt.plot(t_signal, y_signal)
plt.plot(t_idft, np.real(y_idft),'.')
plt.savefig('IDFT', dpi = 600)
plt.show()
