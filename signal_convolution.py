# 该程序用于冲击试验速度时程曲线小波降噪，旨在获得可靠的加速度值用于预测落石冲击力
import numpy as np
import pywt
import matplotlib.pyplot as plt

lower = -5
upper = 5
level = 5

timestep = (upper-lower)/(2**level-1)
t = np.arange(lower, upper+0.5*timestep, timestep)

signal_s = np.array([0,-0.016276871,-0.03410392,-0.055031325,-0.07595873,-0.099211403,-0.11858863,-0.137965857,-0.157343084,-0.175170133,-0.19454736,-0.213924587,-0.231751636,-0.243377972,-0.27050609,-0.292983673,-0.309260544,-0.334063395,-0.353440622,-0.370492581,-0.393745254,-0.413122481,-0.432499708,-0.451876935,-0.47280434,-0.494506835,-0.516984418,-0.534811467,-0.554188694,-0.580541723,-0.598368772,-0.621621444,-0.635573048,-0.654950275,-0.670452056,-0.693704729,-0.724708292,-0.753386588,-0.774313994,-0.793691221,-0.808417913,-0.830120407,-0.850272724,-0.871975218,-0.892902623,-0.915380207,-0.933982345,-0.95490975,-0.973511888,-0.994439293,-1.015366698,-1.038619371,-1.061872043,-1.082799448,-1.100626497,-1.120778813,-1.137055684,-1.158758178,-1.177360316,-1.197512633,-1.222315483,-1.244017978,-1.264170294,-1.28432261,-1.310675639,-1.330827955,-1.350980271,-1.372682765,-1.392059992,-1.409887041,-1.430039357,-1.450191674,-1.47034399,-1.491271395,-1.509873533,-1.534676384,-1.5548287,-1.572655749,-1.591257887,-1.609084935,-1.627687073,-1.645514122,-1.666441528,-1.68116822,-1.699770358,-1.713721962,-1.733874278,-1.750151149,-1.764877841,-1.781154712,-1.798981761,-1.811383186,-1.8245597,-1.840836571,-1.840836571,-1.856338353,-1.873390313,-1.889667183,-1.898193163,-1.903618787,-1.9098195,-1.913694945,-1.915245123,-1.91757039,-1.920670747,-1.920670747,-1.920670747,-1.920670747,-1.919120569,-1.916020212,-1.9098195,-1.902843698,-1.896642985,-1.889667183,-1.88424156,-1.879591025,-1.871840134,-1.861763976,-1.853237997,-1.847037284,-1.838511304,-1.828435146,-1.819134077,-1.810608097,-1.800531939,-1.792781048,-1.784255068,-1.777279266,-1.768753286,-1.761002396,-1.751701327,-1.744725525,-1.734649367,-1.727673565,-1.716047229,-1.70674616,-1.69822018,-1.693569645,-1.684268576,-1.676517686,-1.666441528,-1.65636537,-1.64783939,-1.63931341,-1.628462163,-1.620711272,-1.612185292,-1.603659312,-1.59668351,-1.587382441,-1.576531194,-1.568005214,-1.557929056,-1.553278522,-1.545527631,-1.540102007,-1.533901294,-1.524600225,-1.517624424,-1.508323355,-1.499022286,-1.492046484,-1.483520504,-1.477319791,-1.471119079,-1.464143277,-1.458717653,-1.452516941,-1.44476605,-1.437015159,-1.430814446,-1.425388823,-1.416087754,-1.409887041,-1.40213615,-1.39438526,-1.386634369,-1.376558211,-1.367257142,-1.358731162,-1.350980271,-1.346329736,-1.337028667,-1.326952509,-1.319976708,-1.313000906,-1.307575282,-1.30137457,-1.295948946,-1.288973144,-1.281997343,-1.275021541,-1.269595917,-1.263395205,-1.257194492,-1.248668512,-1.240142532,-1.23316673,-1.226190929,-1.220765305,-1.213789503,-1.206813702,-1.202163167,-1.195962454,-1.19131192,-1.181235762,-1.17425996,-1.168834337,-1.164183802,-1.155657822,-1.147906931,-1.141706219,-1.137830773,-1.134730417,-1.129304793,-1.126204437,-1.120778813,-1.113027923,-1.106052121,-1.100626497,-1.093650696,-1.086674894,-1.082024359,-1.077373825,-1.074273468,-1.070398023,-1.067297667,-1.062647132,-1.057996598,-1.052570974,-1.046370262,-1.040169549,-1.033193747,-1.025442856,-1.019242144,-1.01381652,-1.008390896,-1.002965273,-0.995989471,-0.990563848,-0.986688402,-0.980487689,-0.976612244,-0.97118662,-0.966536086,-0.96343573,-0.959560284,-0.95490975,-0.951034304,-0.945608681,-0.942508324,-0.93785779,-0.931657077,-0.927006543,-0.923906186,-0.919255652,-0.914605117,-0.908404405,-0.905304048,-0.902203692,-0.898328247,-0.892127534,-0.887477,-0.884376643,-0.878175931,-0.874300485,-0.869649951,-0.866549594,-0.862674149,-0.860348882,-0.856473436,-0.85337308,-0.849497635,-0.8448471,-0.842521833,-0.837871298,-0.834770942,-0.831670586,-0.82779514,-0.823919695,-0.81926916,-0.816168804,-0.813068448,-0.811518269,-0.807642824,-0.805317557,-0.801442111,-0.799116844,-0.796791577,-0.792916132,-0.789040686,-0.787490508,-0.783615063,-0.781289795,-0.77741435,-0.773538904,-0.76888837,-0.767338192,-0.764237835,-0.762687657,-0.76036239,-0.756486945,-0.754161677,-0.752611499,-0.752611499,-0.751061321,-0.747960965,-0.746410787,-0.744860608,-0.742535341,-0.742535341,-0.740985163,-0.738659896,-0.737884807,-0.73478445,-0.734009361,-0.731684094,-0.729358827,-0.729358827,-0.72625847,-0.723933203,-0.722383025,-0.720832847,-0.719282669,-0.71850758,-0.71773249,-0.715407223,-0.715407223,-0.713857045,-0.711531778,-0.7099816,-0.709206511,-0.708431421,-0.706106154,-0.703005798,-0.703005798,-0.703005798,-0.70145562,-0.700680531,-0.700680531,-0.700680531,-0.699130353,-0.698355263,-0.696805085,-0.695254907,-0.695254907,-0.693704729,-0.694479818,-0.695254907,-0.694479818,-0.696805085,-0.695254907,-0.697580174,-0.696805085,-0.695254907,-0.695254907,-0.695254907,-0.697580174,-0.694479818,-0.694479818,-0.69292964,-0.696805085,-0.695254907,-0.694479818,-0.695254907,-0.695254907,-0.695254907,-0.695254907,-0.695254907,-0.695254907,-0.696029996,-0.695254907,-0.69292964,-0.695254907,-0.696805085,-0.696805085,-0.696805085,-0.696805085,-0.698355263,-0.696805085,-0.697580174,-0.696029996,-0.699130353,-0.699905442,-0.699905442,-0.699905442,-0.699905442,-0.699905442,-0.699130353,-0.699905442,-0.702230709,-0.703005798,-0.705331065,-0.707656332,-0.710756689,-0.711531778,-0.711531778,-0.713857045,-0.714632134,-0.713857045,-0.713857045,-0.713857045,-0.713857045,-0.716182312,-0.71850758,-0.720832847,-0.720832847,-0.723933203,-0.725483381,-0.727808649,-0.727808649,-0.729358827,-0.729358827,-0.730909005,-0.734009361,-0.736334628,-0.737109718,-0.737884807,-0.740210074,-0.741760252,-0.742535341,-0.744860608,-0.748736054,-0.75183641,-0.753386588,-0.756486945,-0.758812212,-0.761137479,-0.764237835,-0.765788014,-0.768113281,-0.771213637,-0.774313994,-0.776639261,-0.778964528,-0.781289795,-0.783615063,-0.787490508,-0.790590864,-0.79446631,-0.798341755,-0.799891933,-0.803767379,-0.806092646,-0.807642824,-0.812293359,-0.815393715,-0.816943893,-0.81926916,-0.823919695,-0.827020051,-0.830120407,-0.832445675,-0.835546031,-0.837096209,-0.840196566,-0.8448471,-0.848722545,-0.85337308,-0.856473436,-0.859573793,-0.864224327,-0.867324683,-0.871975218,-0.875075574,-0.878175931,-0.882051376,-0.88670191,-0.889802267,-0.894452801,-0.898328247,-0.902203692,-0.906854227,-0.91227985,-0.916930385,-0.923906186,-0.932432166,-0.93785779,-0.944833592,-0.952584483,-0.960335373,-0.965760997,-0.970411531,-0.972736799,-0.975837155,-0.978937511,-0.981262779,-0.982037868,-0.983588046,-0.989788758,-0.994439293,-0.999089827,-1.002965273,-1.007615807,-1.012266342,-1.017691965,-1.0223425,-1.026217945,-1.031643569,-1.036294103,-1.042494816,-1.04792044,-1.054121152,-1.061872043,-1.067297667,-1.071948201,-1.076598736,-1.08124927,-1.086674894,-1.091325428,-1.09830123,-1.102951765,-1.109927566,-1.117678457,-1.12387917,-1.129304793,-1.134730417,-1.138605862,-1.141706219,-1.146356753,-1.150232199,-1.154107644,-1.159533268,-1.164183802,-1.169609426,-1.176585227,-1.18278594,-1.189761742,-1.198287722,-1.206038613,-1.213789503,-1.220765305,-1.228516196,-1.236267087,-1.243242888,-1.250993779,-1.257969581,-1.264170294,-1.271146095,-1.277346808,-1.286647877,-1.293623679,-1.299824392,-1.304474926,-1.310675639,-1.320751797,-1.328502688,-1.337803757,-1.344779558,-1.350980271,-1.357180984,-1.366482053,-1.374232943,-1.381983834,-1.388959636,-1.392835081,-1.399035794,-1.407561774,-1.416862843,-1.424613734,-1.433914803,-1.443990961,-1.451741852,-1.457942564,-1.467243633,-1.475769613,-1.481970326,-1.488171039,-1.494371751,-1.504447909,-1.5121988,-1.52072478,-1.526150404,-1.534676384,-1.542427274,-1.550953254,-1.560254323,-1.568005214,-1.575756105,-1.580406639,-1.587382441,-1.594358243,-1.600558956,-1.608309846,-1.616835826,-1.624586717,-1.636213053,-1.647064301,-1.654040102,-1.660240815,-1.669541884,-1.669541884,-1.681943309,-1.692019467,-1.701320536,-1.709071427,-1.71527214,-1.726123387,-1.736199545,-1.743950436,-1.751701327,-1.759452218,-1.767978197,-1.771853643,-1.777279266,-1.789680692,-1.795106315,-1.801307028,-1.809057919,-1.812933364,-1.820684255,-1.828435146,-1.836186037,-1.84161166,-1.846262195,-1.847812373,-1.851687818,-1.857113442,-1.857888531,-1.864864333,-1.869514867,-1.873390313,-1.876490669,-1.877265758,-1.880366114,-1.880366114,-1.880366114,-1.880366114,-1.880366114,-1.877265758,-1.874940491,-1.872615224,-1.871065045,-1.8671896,-1.863314155,-1.861763976,-1.859438709,-1.851687818,-1.846262195,-1.843161838,-1.835410948,-1.828435146,-1.823009522,-1.814483542,-1.80828283,-1.801307028,-1.795106315,-1.792005959,-1.786580335,-1.780379623,-1.773403821,-1.767203108,-1.762552574,-1.756351861,-1.749376059,-1.742400258,-1.736199545,-1.729223743,-1.722247942,-1.714497051,-1.710621605,-1.70674616,-1.702870714,-1.695119824,-1.688919111,-1.684268576,-1.677292775,-1.671092062,-1.66411626,-1.660240815,-1.653265013,-1.64783939,-1.641638677,-1.636213053,-1.633112697,-1.626136895,-1.620711272,-1.616060737,-1.609860025,-1.604434401,-1.599008777,-1.592032976,-1.585057174,-1.57963155,-1.574205927,-1.57110557,-1.567230125,-1.561029412,-1.554053611,-1.547077809,-1.541652185,-1.539326918,-1.536226562,-1.530800938,-1.526925493,-1.521499869,-1.515299156,-1.509098444,-1.505222998,-1.499797375,-1.496697019,-1.492046484,-1.48739595,-1.482745415,-1.47886997,-1.475769613,-1.474219435,-1.469568901,-1.467243633,-1.464918366,-1.460267832,-1.455617297,-1.450191674,-1.445541139,-1.440115515,-1.434689892,-1.430039357,-1.426163912,-1.423063556,-1.418413021,-1.414537576,-1.409887041,-1.408336863,-1.406011596,-1.401361061,-1.398260705,-1.392835081,-1.388959636,-1.385084191,-1.378883478,-1.375783122,-1.371132587,-1.365706964,-1.36028134,-1.357956073,-1.357956073,-1.355630805,-1.353305538,-1.349430093,-1.346329736,-1.341679202,-1.338578846,-1.336253578,-1.331603044,-1.328502688,-1.324627242,-1.320751797,-1.31765144,-1.31765144,-1.313775995,-1.312225817,-1.309125461,-1.306025104,-1.302924748,-1.299049302,-1.297499124,-1.295173857,-1.292073501,-1.290523323,-1.289748233,-1.287422966,-1.287422966,-1.28432261,-1.281997343,-1.281997343,-1.278121897,-1.273471363,-1.271921185,-1.269595917,-1.268045739,-1.268045739,-1.268820828,-1.265720472,-1.263395205,-1.261845026,-1.261845026,-1.260294848,-1.259519759,-1.25874467,-1.256419403,-1.253319047,-1.252543957,-1.25021869,-1.25021869,-1.247893423,-1.246343245,-1.246343245,-1.246343245,-1.242467799,-1.242467799,-1.242467799,-1.240142532,-1.240142532,-1.237817265,-1.237817265,-1.237817265,-1.237817265,-1.237817265,-1.235491998,-1.235491998,-1.235491998,-1.235491998,-1.23316673,-1.231616552,-1.231616552,-1.230841463,-1.229291285,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.230066374,-1.230841463,-1.231616552,-1.23316673,-1.234716909,-1.236267087,-1.237817265,-1.237817265,-1.240142532,-1.24169271,-1.243242888,-1.245568156,-1.247118334,-1.250993779,-1.254094136,-1.255644314,-1.257194492,-1.257194492,-1.259519759,-1.261845026,-1.262620116,-1.263395205,-1.263395205,-1.264945383,-1.26727065,-1.268820828,-1.270371006,-1.272696274,-1.274246452,-1.27579663,-1.278896986,])
signal_handle = signal_s

scale = 1
s = scale

C = 1/np.sqrt(np.pi)
theta_t = C * np.exp(-t**2)
theta_st = C/np.sqrt(s) * np.exp(-t**2/(s**2))

C1 = np.sqrt(np.sqrt(2/np.pi))
C2 = np.sqrt(np.sqrt(2/(9*np.pi)))

psi_1st = C1 * np.exp(-t**2)*(-2*t)
psi_1st_st = C1 * np.exp(-t**2/s**2) * (2*t/s) / (np.sqrt(s))
psi_2nd = C2 * np.exp(-t**2) * (4*t**2 - 2)
'''
signal_noise = np.sin(t)+np.random.randn(10)
c = np.zeros(10)+0.1
miu = 0
sigma = 1
gauss_array = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(t-miu)**2/(2*sigma**2))
'''
wavelet_gaus1 = pywt.ContinuousWavelet('gaus1')
wavelet_gaus2 = pywt.ContinuousWavelet('gaus2')
wavelet_gaus3 = pywt.ContinuousWavelet('gaus3')
wavelet_mexh = pywt.ContinuousWavelet('mexh')

psi_g1, x1 = wavelet_gaus1.wavefun(level = level)
psi_g2, x2 = wavelet_gaus2.wavefun(level = level)
psi_g3, x3 = wavelet_gaus3.wavefun(level = level)
psi_g4, x4 = wavelet_mexh.wavefun(level = level)

myconv = np.convolve(signal_handle, psi_1st_st, 'same')

scale_use = np.array([scale])
sampling_period=0.002
cwtmatr, freqs = pywt.cwt(signal_handle, scale_use, 'gaus1', method='conv')  #小波分解
# f = pywt.scale2frequency('gaus1', scale_use)/0.002
f = pywt.central_frequency('gaus1')
'''
plt.plot(t, psi_2nd,'-*', label = 'psi_2nd')
plt.plot(t2, psi_t2,label = 'psi_t2')
plt.plot(t3, psi_t3,label = 'psi_t3')
#plt.plot(t4, psi_t4,label = 'psi_t4')

'''
#plt.plot(signal_s)
plt.plot(psi_1st,'-o', label = 'psi_1st')
plt.plot(psi_g1, label = 'psi_t1')
plt.plot(cwtmatr[0,:],label = 'cwtmatr[0,:]')
plt.plot(myconv,label = 'myconv')

plt.legend(loc="best",fontsize=8)
plt.show()