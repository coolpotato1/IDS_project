import matplotlib.pyplot as plt
import numpy as np
names = ["KDD_filtered", "MitM", "UDP_DOS", "SF"]
# accuracies_nn = [0.77, 0.85, 0.72, 0.9]
# accuracies_rf = [0.66, 0.55, 0.87, 0.54]

# The below results were for the individual datasets where each dataset was trained and evaluated on 4 times. This was the basis for dataset_complexity.png
Precisions_nn = [[0.9664978125719549, 0.9653855059041445, 0.9648235429359696, 0.9659667743424089, 0.9655771905424201], [0.9453416149068323, 0.9532828282828283, 0.9453416149068323, 0.9498117942283564, 0.942998760842627], [0.9623115577889447, 0.9623115577889447, 0.9623115577889447, 0.9623115577889447, 0.9623115577889447], [0.9977477477477478, 0.9977477477477478, 0.9977477477477478, 0.9977477477477478, 0.9977477477477478]]
Recalls_nn = [[0.8496103633235502, 0.8439429207570084, 0.8494079546604595, 0.8473838680295517, 0.8431332861046453], [0.9465174129353234, 0.9390547263681592, 0.9465174129353234, 0.9415422885572139, 0.9465174129353234], [0.9960988296488946, 0.9960988296488946, 0.9960988296488946, 0.9960988296488946, 0.9960988296488946], [1.0, 1.0, 1.0, 1.0, 1.0]]
Fscores_nn = [[0.9042925620724942, 0.9005885846967978, 0.9034445640473627, 0.9027979945010512, 0.9002107083040684], [0.9459291485394655, 0.9461152882205514, 0.9459291485394655, 0.9456589631480325, 0.9447548106765984], [0.9789137380191693, 0.9789137380191693, 0.9789137380191693, 0.9789137380191693, 0.9789137380191693], [0.9988726042841037, 0.9988726042841037, 0.9988726042841037, 0.9988726042841037, 0.9988726042841037]]
Accuracies_nn = [[0.9092997141690486, 0.9060330747243773, 0.9084320130665577, 0.9079726418946509, 0.9057268272764394], [0.943579766536965, 0.9442282749675746, 0.943579766536965, 0.943579766536965, 0.9422827496757458], [0.9974111555660156, 0.9974111555660156, 0.9974111555660156, 0.9974111555660156, 0.9974111555660156], [0.9992641648270787, 0.9992641648270787, 0.9992641648270787, 0.9992641648270787, 0.9992641648270787]]

Precisions_rf = [[0.9698373114831967, 0.9698337292161521, 0.9686293436293436, 0.9696317551410808, 0.9697257509200997], [0.978021978021978, 0.9792176039119804, 0.9768009768009768, 0.9792176039119804, 0.9768292682926829], [0.8428571428571429, 0.6742756804214223, 0.6736842105263158, 0.6742756804214223, 0.6742756804214223], [0.9977477477477478, 0.9977477477477478, 0.9977477477477478, 0.9977477477477478, 0.9977477477477478]]
Recalls_rf = [[0.8265357757312013, 0.8264345713996559, 0.812468373646392, 0.8207671288331141, 0.8266369800627467], [0.996268656716418, 0.996268656716418, 0.9950248756218906, 0.996268656716418, 0.996268656716418], [0.9973992197659298, 0.9986996098829649, 0.9986996098829649, 0.9986996098829649, 0.9986996098829649], [1.0, 1.0, 1.0, 1.0, 1.0]]
Fscores_rf = [[0.8924707682220523, 0.8924102508059669, 0.8837030106224888, 0.8890106878596876, 0.8924825174825175], [0.9870609981515712, 0.9876695437731196, 0.9858287122612446, 0.9876695437731196, 0.9864532019704433], [0.9136390708755212, 0.8050314465408804, 0.8046097433211105, 0.8050314465408804, 0.8050314465408804], [0.9988726042841037, 0.9988726042841037, 0.9988726042841037, 0.9988726042841037, 0.9988726042841037]]
Accuracies_rf = [[0.8995508370763577, 0.8994997958350347, 0.8921498570845243, 0.8966414863209473, 0.8995508370763577], [0.9863813229571985, 0.9870298313878081, 0.9850843060959793, 0.9870298313878081, 0.9857328145265889], [0.9886247744567349, 0.9708166627441751, 0.9707382129128422, 0.9708166627441751, 0.9708166627441751], [0.9992641648270787, 0.9992641648270787, 0.9992641648270787, 0.9992641648270787, 0.9992641648270787]]


# Precisions_rf = [[0.9473684210526315, 0.9461531523405791, 0.9224764468371467, 0.9452852153667055, 0.9421317064417728, 0.9236614242799082, 0.9445218845832172, 0.9468066020810908, 0.9244058581075155, 0.9246821569487067]]
# Recalls_rf = [[0.8591572123176662, 0.8500810372771475, 0.8331442463533225, 0.8554294975688816, 0.8509724473257698, 0.8471636952998379, 0.8236628849270664, 0.8553484602917342, 0.8542139384116694, 0.8546191247974068]]
# Fscores_rf = [[0.9011091751306787, 0.8955478721133735, 0.8755375771769214, 0.8981154549708598, 0.8942348633228305, 0.8837602502324795, 0.8799619064109778, 0.8987568119891007, 0.8879248620646086, 0.8882712149926301]]
# Accuracies_rf = [[0.9364190278422908, 0.9331402497335993, 0.9201344299024563, 0.934561053580699, 0.9321292931500861, 0.9248613350091532, 0.9242329025383207, 0.9350255471460969, 0.9272930954397661, 0.9275116806470122]]
#
# Precisions_nn = [[0.9327225130890052, 0.9450192914766748, 0.9395157765929552, 0.9391447368421053, 0.9452472144846796, 0.8717054577893115, 0.9189794862954663, 0.9436243997865907, 0.8791091387245233, 0.8957214765100671]]
# Recalls_nn = [[0.8662074554294976, 0.8733387358184765, 0.8710696920583468, 0.8791734197730956, 0.8799837925445705, 0.8710696920583468, 0.8640194489465154, 0.859967585089141, 0.8668557536466774, 0.8652350081037277]]
# Fscores_nn = [[0.898235294117647, 0.9077661725067385, 0.9039989907909677, 0.9081700987778335, 0.9114487157965419, 0.8713874589599124, 0.8906524099908111, 0.8998558466887137, 0.8729394483433981, 0.8802143446001648]]
# Accuracies_nn = [[0.9338233285062434, 0.9401622995163802, 0.9376212464821443, 0.9400530069127572, 0.9423481515888412, 0.9133036421760158, 0.9284679909287139, 0.9354627175605891, 0.9149157080794558, 0.9205989234678543]]
precision_nn = np.mean(Precisions_nn, axis=1)
recall_nn = np.mean(Recalls_nn, axis=1)
fscore_nn = np.mean(Fscores_nn, axis=1)
accuracy_nn = np.mean(Accuracies_nn, axis=1)

precision_rf = np.mean(Precisions_rf, axis=1)
recall_rf = np.mean(Recalls_rf, axis=1)
fscore_rf = np.mean(Fscores_rf, axis=1)
accuracy_rf = np.mean(Accuracies_rf, axis=1)

index = np.arange(len(precision_nn))
width = 0.3

plt.title("Test")
plt.subplot(221)
plt.bar(index - width, accuracy_nn, width=width, color='C1', align='center', label='DNN')
plt.bar(index, accuracy_rf, width=width, color='C2', align='center', label='RF')
plt.ylabel("Accuracy")
plt.xticks(index - width/2, names)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Dataset accuracies for RF and DNN")

plt.subplot(222)
plt.bar(index - width, recall_nn, width=width, color='C1', align='center', label='DNN')
plt.bar(index, recall_rf, width=width, color='C2', align='center', label='RF')
plt.ylabel("Recall")
plt.xticks(index - width/2, names)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Dataset recalls for RF and DNN")

plt.subplot(223)
plt.bar(index - width, precision_nn, width=width, color='C1', align='center', label='DNN')
plt.bar(index, precision_rf, width=width, color='C2', align='center', label='RF')
plt.ylabel("Precision")
plt.xticks(index - width/2, names)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Precision on datasets for RF and DNN")

plt.subplot(224)
plt.bar(index - width, fscore_nn, width=width, color='C1', align='center', label='DNN')
plt.bar(index, fscore_rf, width=width, color='C2', align='center', label='RF')
plt.ylabel("F1-score")
plt.xticks(index - width/2, names)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("F1-score on the datasets for RF and DNN")

plt.show()