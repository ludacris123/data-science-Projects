import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "engine", "label": "Model engine", "options": ["baseline", "advanced"], "default": "baseline"}, {"key": "house", "label": "House features (JSON)", "type": "object", "default": {"bedrooms": 3, "area": 1500, "age": 5}}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'real-estate-predictor',Controls});
