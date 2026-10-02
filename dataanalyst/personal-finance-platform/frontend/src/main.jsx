import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "budgets", "label": "Monthly category budgets (JSON)", "type": "object", "default": {}}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'personal-finance-platform',Controls});
