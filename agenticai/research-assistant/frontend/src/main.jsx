import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "query", "label": "Research question"}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'research-assistant',Controls});
