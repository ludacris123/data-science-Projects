import argparse,importlib,json,sys
from pathlib import Path
import uvicorn
p=argparse.ArgumentParser();p.add_argument('project');p.add_argument('--port',type=int,default=8000);args=p.parse_args()
paths=list(Path(__file__).parent.glob('*/'+args.project+'/project.json'))
if len(paths)!=1:p.error('Unknown project; see PROJECTS.md')
from portfolio_core.app import create_app
uvicorn.run(create_app(args.project,json.loads(paths[0].read_text())),host='127.0.0.1',port=args.port)
