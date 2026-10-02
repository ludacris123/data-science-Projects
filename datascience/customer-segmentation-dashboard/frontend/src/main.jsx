import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "algorithm", "label": "Clustering method", "options": ["kmeans", "dbscan"], "default": "kmeans"}, {"key": "clusters", "label": "Number of segments", "type": "number", "min": 2, "max": 20, "default": 3}, {"key": "eps", "label": "DBSCAN radius", "type": "number", "min": 0.1, "default": 1}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'customer-segmentation-dashboard',Controls});
