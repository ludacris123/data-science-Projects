import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "horizon", "label": "Forecast days", "type": "number", "min": 1, "max": 60, "default": 7}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'disease-outbreak-predictor',Controls});
