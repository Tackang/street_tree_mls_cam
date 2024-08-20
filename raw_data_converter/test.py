import yaml

# Load the folderList from the YAML file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

folderList = config['folderList']
rootPath = config['rootPath']

print(rootPath)
print(folderList)