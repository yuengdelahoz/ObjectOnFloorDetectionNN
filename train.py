from Network.Net import Network

n = Network()
n.initialize('topology_04')
n.train()
n.evaluate()

n.initialize('topology_03')
n.train()
n.evaluate()

n.initialize('topology_0')
n.train()
n.evaluate()

n.initialize('topology_05')
n.train()
n.evaluate()

n.initialize('topology_01')
n.train()
n.evaluate()
