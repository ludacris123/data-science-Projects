import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "query", "label": "Text search (CLIP)"}, {"key": "encoder", "label": "Image encoder", "options": ["efficientnet", "clip"], "default": "efficientnet"}, {"key": "catalog", "label": "Catalog images (JSON)", "type": "object", "default": []}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'visual-search-engine',Controls});
