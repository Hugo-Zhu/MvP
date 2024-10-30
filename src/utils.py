import pickle

def load_data(file_path):
    data = pickle.load(open(file_path, 'rb'))
    text = data['posts_text']
    label = data['annotations']
    processed_data = process_data(text, label)
    return processed_data

def process_data(poster, label):
    label_lookup = {'E': 1, 'I': 0, 'S': 1, 'N':0, 'T': 1, 'F': 0, 'J': 1, 'P':0}
    poster_data = [{'posts': t, 
                'label0': label_lookup[list(label[i])[0]],
                'label1': label_lookup[list(label[i])[1]],
                'label2': label_lookup[list(label[i])[2]],
                'label3': label_lookup[list(label[i])[3]]} 
               for i,t in enumerate(poster)]
    return poster_data
    
def statistical_analysis(label):
    persona_lookup = {}
    I,E,S,N,T,F,P,J=0,0,0,0,0,0,0,0
    for t in label:
        if 'I' in t:
            I+=1
        if 'E' in t:
            E += 1
        if 'S' in t:
            S+=1
        if 'N' in t:
            N+=1
        if 'T' in t:
            T+=1
        if 'F' in t:
            F+=1
        if 'P' in t:
            P+=1
        if 'J' in t:
            J+=1
        if t not in persona_lookup:
            persona_lookup[t] = 1
        else:
            persona_lookup[t] += 1
    print('I', I)
    print('E', E)
    print('S', S)
    print('N', N)
    print('T', T)
    print('F', F)
    print('P', P)
    print('J', J)
    print("persona number:", persona_lookup)