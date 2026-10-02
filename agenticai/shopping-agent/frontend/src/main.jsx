import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "query", "label": "Budget and specifications"}, {"key": "product_urls", "label": "Optional product page URLs (JSON)", "type": "object", "default": []}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'shopping-agent',Controls});
