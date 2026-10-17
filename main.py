import numpy as np
import network
import pygame
import math
#import torchvision
#import torchvision.transforms as transforms

pygame.init()
screen = pygame.display.set_mode((280, 280))
clock = pygame.time.Clock()
running = True
#data setup


'''
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
batches = train_dataset.data.numpy().astype(np.float32)/255.0
batches_t = test_dataset.data.numpy().astype(np.float32)/255.0
batches = np.transpose(np.expand_dims(batches, axis=3), (1, 2, 3, 0))
batches_t = np.transpose(np.expand_dims(batches_t, axis=3), (1, 2, 3, 0))
batches =  np.split(batches, 1875, axis=3)
batches_t = np.split(batches_t, 10000, axis=3)
correct = np.hsplit(train_dataset.targets.numpy(), 1875)
correct_t = np.hsplit(test_dataset.targets.numpy(), 10000)
'''

weights = np.load('C:/Users/janan/Coding!/NumberPrediction-CNN/weights.npz')
Ws = [weights['W0'], weights['W1'], weights['W2'], weights['W3']]
bs = [weights['b0'], weights['b1'], weights['b2'], weights['b3']]
MLP = network.MLP(Ws, bs)
#print(MLP.predict(test1))
#training

prev_pos = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            prev_pos = None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                surf = pygame.transform.smoothscale(screen, (28, 28))
                pixels = pygame.surfarray.array3d(surf)
                x = pixels[:, :, 0].T
                x = x/255
                
                nonzero = np.nonzero(x)
                min_row = min(nonzero[0])
                max_row = max(nonzero[0])
                min_col = min(nonzero[1])
                max_col = max(nonzero[1])
                cropped = x[min_row:max_row+1, min_col:max_col+1]
                cropped_u8 = (cropped*255).astype(np.uint8)
                surf = pygame.surfarray.make_surface(np.stack([cropped_u8.T]*3, axis=-1))
                h, w = cropped.shape
                if h > w:
                    w = int(20*w/h)
                    h = 20
                else:
                    h = int(20*h/w)
                    w = 20
                resized_surf  = pygame.transform.smoothscale(surf, (w, h))
                resized_array = pygame.surfarray.array3d(resized_surf)[:, :, 0]/255
                x = np.zeros((28, 28))
                start_row = (28 - h)//2
                start_col = (28 - w)//2
                x[start_row:start_row + h, start_col:start_col+w] = resized_array.T
                X = x.reshape(28, 28, 1, 1)
                print(MLP.predict(X))
                #plt.imshow(x, cmap='gray')
                #plt.show()
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_c:
                screen.fill((0,0,0))

    if pygame.mouse.get_pressed()[0]:
        curr_pos = np.array(pygame.mouse.get_pos())
        if prev_pos is not None:
            dist = curr_pos - prev_pos
            steps = np.linalg.norm(dist)/20
            steps = math.ceil(steps) + 3
            for i in range(0, int(steps)):
                pygame.draw.circle(screen, 'white', tuple(prev_pos + ((i+1)/steps)*dist), 10)
        else:
            pygame.draw.circle(screen, 'white', tuple(curr_pos), 10)
        prev_pos = curr_pos

    pygame.display.flip()
    clock.tick(60)
'''
epoch_n = 7
MLP = network.MLP()
for i in range(0, epoch_n):
    for ind, (batch, c) in enumerate(zip(batches, correct)):
        step = i * 1875 + ind
        alpha = 0.001*(1 - (step/13125))
        MLP.forward(batch)
        J, G_out = MLP.loss(c)
        MLP.backward(G_out)
        MLP.update(alpha, step+1)
        print(J)

num_correct = 0
for i in range(0, len(correct_t)):
    if MLP.predict(batches_t[i]) == correct_t[i]:
        num_correct+=1
print(f"accuracy: {num_correct/len(correct_t)}")

Ws, bs, = MLP.save_weight()
np.savez('NumberPrediction-CNN/weights.npz', W0=Ws[0], b0=bs[0], W1=Ws[1], b1=bs[1], W2=Ws[2], b2=bs[2], W3=Ws[3], b3=bs[3])
'''