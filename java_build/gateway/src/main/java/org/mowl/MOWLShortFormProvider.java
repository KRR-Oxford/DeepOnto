package org.mowl;

import org.semanticweb.owlapi.util.ShortFormProvider;
import org.semanticweb.owlapi.model.OWLEntity;

public class MOWLShortFormProvider implements ShortFormProvider {

    @Override
    public String getShortForm(OWLEntity entity) {
        return entity.toString();
    }

    @Override
    public void dispose() {}
    
}
