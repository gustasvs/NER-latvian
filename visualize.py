import matplotlib.pyplot as plt
import numpy as np

tiny_bert_validation = [{'loss': 0.48166557344106525, 'precision': 0.3899430740037951, 'recall': 0.4076369947929581, 'f1': 0.39859376894168996}, {'loss': 0.3777653618500783, 'precision': 0.5226489414081733, 'recall': 0.5264071410860401, 'f1': 0.524521309450278}, {'loss': 0.32683200967999604, 'precision': 0.5639466916062661, 'recall': 0.598065955864121, 'f1': 0.5805054151624549}, {'loss': 0.2997004579179562, 'precision': 0.6030977734753146, 'recall': 0.6179023059757005, 'f1': 0.6104102878138394}, {'loss': 0.28040197319709337, 'precision': 0.6421237165038818, 'recall': 0.635755021076122, 'f1': 0.6389234986294543}, {'loss': 0.269110544398427, 'precision': 0.622524182404422, 'recall': 0.6702206793949913, 'f1': 0.6454925373134328}, {'loss': 0.2576463236831702, 'precision': 0.6596669080376538, 'recall': 0.6776593106868336, 'f1': 0.6685420743639922}, {'loss': 0.24989463881804394, 'precision': 0.6701080432172869, 'recall': 0.6920406645177287, 'f1': 0.6808977799463283}, {'loss': 0.24271240655619364, 'precision': 0.6643573817486861, 'recall': 0.6895611207537813, 'f1': 0.6767246623676846}, {'loss': 0.23796254340559245, 'precision': 0.6879789272030651, 'recall': 0.7123729233820977, 'f1': 0.6999634547447923}] 
tiny_bert_training = [{'loss': 0.6860848182630935, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.4827543886250102, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.4163590249674054, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.37757999601280606, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.34834982929065506, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.3245537820451333, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.3045337374926038, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.2873836952237293, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.27227782002360407, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.25875438322841837, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}]

tiny_bert_without_translations_validation = [{'loss': 0.48050981192634656, 'precision': 0.37321135350692003, 'recall': 0.3944954128440367, 'f1': 0.3835583413693347}, {'loss': 0.3788900937598485, 'precision': 0.48826066123622425, 'recall': 0.505331019092487, 'f1': 0.4966492019008164}, {'loss': 0.32572974771834334, 'precision': 0.5631366974650557, 'recall': 0.589387552690305, 'f1': 0.5759631693724255}, {'loss': 0.2955289083604629, 'precision': 0.6034739454094293, 'recall': 0.6030250433920159, 'f1': 0.6032494108892472}, {'loss': 0.2814981783238741, 'precision': 0.6432264736297828, 'recall': 0.6169104884701215, 'f1': 0.6297936970003798}, {'loss': 0.26485707046320806, 'precision': 0.6405405405405405, 'recall': 0.6464170592610959, 'f1': 0.6434653831914106}, {'loss': 0.2546041081444575, 'precision': 0.66775, 'recall': 0.6622861393503595, 'f1': 0.6650068467571268}, {'loss': 0.2548028196279819, 'precision': 0.6585307489830103, 'recall': 0.6823704438383338, 'f1': 0.6702386751095958}]
tiny_bert_without_translations_training = [{'loss': 0.7249792290700449, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.4814681966562529, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.3981902882830747, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.35333777303918906, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.3205189186806384, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.2945717494704663, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.27384447302798737, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.25617428912878265, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}]

distilbert_base_uncased_validation = [{'loss': 0.13809755415870592, 'precision': 0.7741399762752076, 'recall': 0.8090751301760476, 'f1': 0.7912221144519883}, {'loss': 0.12157936363409345, 'precision': 0.8585, 'recall': 0.8514753285395488, 'f1': 0.8549732354039588}, {'loss': 0.10316489762984789, 'precision': 0.8721674876847291, 'recall': 0.8780064468137863, 'f1': 0.8750772272334116}, {'loss': 0.12128613854830082, 'precision': 0.8567237163814181, 'recall': 0.8688321348871808, 'f1': 0.862735442570479}]
distilbert_base_uncased_training = [{'loss': 0.3006931657940503, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.15941120926600455, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.11271655820022534, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.08304800753284804, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}]

distilbert_base_uncased_without_translations_validation = [{'loss': 0.30089355613604823, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.13636895591522563, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.08649345041831603, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.0570779299991867, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}]
distilbert_base_uncased_without_translations_training = [{'loss': 0.13684672207499926, 'precision': 0.8145662440964454, 'recall': 0.812546491445574, 'f1': 0.8135551142005958}, {'loss': 0.11019404456019402, 'precision': 0.8683544303797468, 
'recall': 0.8504835110339698, 'f1': 0.8593260678942755}, {'loss': 0.10760784096156176, 'precision': 0.8670082967301123, 'recall': 0.8809818993305232, 'f1': 0.8739392448653303}, {'loss': 0.10976646994956984, 'precision': 0.8758331276228092, 'recall': 0.8797421274485495, 'f1': 0.8777832756061356}]

distilbert_base_multilingual_cased_validation = [{'loss': 0.09190530994763742, 'precision': 0.8606001936108422, 'recall': 0.8797624938149431, 'f1': 0.870075850256912}, {'loss': 0.08706519572923963, 'precision': 0.8776350860189, 'recall': 0.8960910440376052, 'f1': 0.8867670461500796}, {'loss': 0.08289522254624619, 'precision': 0.8895645828265629, 'recall': 0.904750123701138, 'f1': 0.8970930945664173}, {'loss': 0.09046036740406775, 
'precision': 0.8816524908869988, 'recall': 0.8975754576942108, 'f1': 0.8895427240407011}]
distilbert_base_multilingual_cased_training = [{'loss': 0.20721051364477874, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.11467676336883685, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.0810768822539319, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.05966053413232178, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}]

tiny_mamba_validation = [{'loss': 0.8455469782057876, 'precision': 0.09637166442695334, 'recall': 0.12419594260267194, 'f1': 0.10852880769646524}, {'loss': 0.7027750518101062, 'precision': 0.1726344659454518, 'recall': 0.28030677882236515, 'f1': 0.21367279585101367}, {'loss': 0.6304538230120446, 'precision': 0.20686580472581365, 'recall': 0.34438396833250867, 'f1': 0.2584718224863058}, {'loss': 0.5882063136943059, 'precision': 0.2238020759277691, 'recall': 0.38941118258287977, 'f1': 0.2842437923250564}, {'loss': 0.5556178335172329, 'precision': 0.24473721896033224, 'recall': 0.42281048985650665, 'f1': 0.3100226757369614}, {'loss': 0.5218784357475038, 'precision': 0.2694740089464754, 'recall': 0.4322117763483424, 'f1': 0.33197149643705465}, {'loss': 0.508753581137961, 'precision': 0.276355421686747, 'recall': 0.45398317664522514, 'f1': 0.3435686201085939}, {'loss': 0.49919619532050313, 'precision': 0.28630766948523956, 'recall': 0.4774863928748144, 'f1': 0.35797088008902905}, {'loss': 0.48762948019969415, 'precision': 0.31223954313937335, 'recall': 0.5004948045522019, 'f1': 0.3845642049234864}, {'loss': 0.4999896304027454, 'precision': 0.296127562642369, 'recall': 0.5145967342899554, 'f1': 0.3759262606181095}]
tiny_mamba_training =  [{'loss': 1.2884913857181868, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.8300835791627567, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.6898327801773946, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.5989537152349949, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.5329158688733975, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.48324468030283846, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.44196044133603574, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.40598330420007306, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.3751864673842986, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.34738397487252953, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}]

base_mamba_validation =  [{'loss': 0.49836237678253975, 'precision': 0.3136907399202481, 'recall': 0.3503216229589312, 'f1': 0.33099579242636745}, {'loss': 0.40336464408741957, 'precision': 0.3984835720303286, 'recall': 0.46808510638297873, 'f1': 0.43048919226393634}, {'loss': 0.38155225939458975, 'precision': 0.4108543565825737, 'recall': 0.47946561108362196, 'f1': 0.4425162689804772}, {'loss': 0.3492819500842552, 'precision': 0.43711688860705983, 'recall': 0.5116279069767442, 'f1': 0.47144648352900953}, {'loss': 0.33895352026124026, 'precision': 0.4425403225806452, 'recall': 0.5430479960415636, 'f1': 0.4876694067984893}, {'loss': 0.36600572546102594, 'precision': 0.4454621149042465, 'recall': 0.5294408708560119, 'f1': 0.4838345014695908}]
base_mamba_training = [{'loss': 0.6710619447960777, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.45177940773664843, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.3660467555110438, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.3069128007879658, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.2613573901340334, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'loss': 0.22565894060731687, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}]


bg_color = '#212121'
plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['axes.facecolor'] = bg_color
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['legend.facecolor'] = bg_color
plt.rcParams['legend.edgecolor'] = 'white'

def plot_loss(train, val, title):
    epochs_train = list(range(1, len(train)+1))
    epochs_val = list(range(1, len(val)+1))
    plt.figure()
    plt.plot(epochs_train, [d['loss'] for d in train], label='Training Loss', linewidth=2)
    plt.plot(epochs_val, [d['loss'] for d in val], label='Validation Loss', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)
    plt.show()

def plot_metrics(val, title):
    epochs = list(range(1, len(val)+1))
    plt.figure()
    plt.plot(epochs, [d['precision'] for d in val], label='Precision', linewidth=2)
    plt.plot(epochs, [d['recall'] for d in val], label='Recall', linewidth=2)
    plt.plot(epochs, [d['f1'] for d in val], label='F1', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(False)
    plt.show()


# plot_loss(tiny_bert_training, tiny_bert_validation, 'Tiny BERT: train / validation loss')
# plot_loss(tiny_bert_without_translations_training, tiny_bert_without_translations_validation, 'Tiny BERT without translations: train / validation loss')
# plot_loss(distilbert_base_uncased_training, distilbert_base_uncased_validation, 'DistilBERT Uncased: train / validation loss')
# plot_loss(distilbert_base_uncased_without_translations_training, distilbert_base_uncased_without_translations_validation, 'DistilBERT Uncased without translations: train / validation loss')
# plot_loss(distilbert_base_multilingual_cased_training, distilbert_base_multilingual_cased_validation, 'DistilBERT Multilingual: train / validation loss')
# plot_loss(tiny_mamba_training, tiny_mamba_validation, 'Tiny Mamba: train / validation loss')
plot_loss(base_mamba_training, base_mamba_validation, 'Base Mamba: train / validation loss')


# plot_metrics(tiny_bert_validation, 'Tiny BERT: metrics')
# plot_metrics(tiny_bert_without_translations_validation, 'Tiny BERT without translations: metrics')
# plot_metrics(distilbert_base_uncased_validation, 'DistilBERT Uncased: metrics')
# plot_metrics(distilbert_base_uncased_without_translations_validation, 'DistilBERT Uncased without translations: metrics')
# plot_metrics(distilbert_base_multilingual_cased_validation, 'DistilBERT Multilingual: metrics')
# plot_metrics(tiny_mamba_validation, 'Tiny Mamba: metrics')
plot_metrics(base_mamba_validation, 'Base Mamba: metrics')


plt.figure()
epochs_tiny = list(range(1, len(tiny_bert_validation)+1))
epochs_tiny_without_translations = list(range(1, len(tiny_bert_without_translations_validation)+1))
epochs_uncased = list(range(1, len(distilbert_base_uncased_validation)+1))
epochs_uncased_without_translations = list(range(1, len(distilbert_base_uncased_without_translations_validation)+1))
epochs_multi = list(range(1, len(distilbert_base_multilingual_cased_validation)+1))
epochs_tiny_mamba = list(range(1, len(tiny_mamba_validation)+1))
epochs_base_mamba = list(range(1, len(base_mamba_validation)+1))
plt.plot(epochs_tiny, [d['loss'] for d in tiny_bert_validation], label='Tiny BERT', linewidth=2)
plt.plot(epochs_tiny_without_translations, [d['loss'] for d in tiny_bert_without_translations_validation], label='Tiny BERT w/o Translations', linewidth=2)
plt.plot(epochs_uncased, [d['loss'] for d in distilbert_base_uncased_validation], label='DistilBERT Uncased', linewidth=2)
plt.plot(epochs_uncased_without_translations, [d['loss'] for d in distilbert_base_uncased_without_translations_validation], label='DistilBERT Uncased w/o Translations', linewidth=2)
plt.plot(epochs_multi, [d['loss'] for d in distilbert_base_multilingual_cased_validation], label='DistilBERT Multilingual', linewidth=2)
plt.plot(epochs_tiny_mamba, [d['loss'] for d in tiny_mamba_validation], label='Tiny Mamba', linewidth=2)
plt.plot(epochs_base_mamba, [d['loss'] for d in base_mamba_validation], label='Base Mamba', linewidth=2)
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)
plt.show()