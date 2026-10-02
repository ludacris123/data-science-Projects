import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "symbol", "label": "Optional live ticker"}, {"key": "headlines", "label": "Licensed dated headlines (JSON)", "type": "object", "default": []}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'stock-sentiment-analyzer',Controls});
