import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "engine", "label": "Model engine", "options": ["baseline", "advanced"], "default": "baseline"}, {"key": "trials", "label": "Optuna tuning trials", "type": "number", "min": 1, "max": 30, "default": 10}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'customer-churn-saas',Controls});
